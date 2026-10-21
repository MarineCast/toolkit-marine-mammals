"""Conservative source-record delivery for an explicitly supplied study mask.

This adapter deliberately does not consume proximity-merged composite identities.
It uses the toolkit's reviewed species interpretation, without label imputation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import h3
import pandas as pd
import pyarrow as pa
from shapely.geometry import Point
from shapely.prepared import prep

from marine_mammal_toolkit.tools.observations.process.pipeline import (
    _coordinates, _parse_timestamp, _parse_explicit_date, _contains_clock,
)

SCHEMA_VERSION = "orca-study-sightings-v1"
SIGHTINGS_SCHEMA = pa.schema([
    pa.field('event_id',pa.string(),False),
    pa.field('source',pa.string(),False),
    pa.field('source_record_id',pa.string(),False),
    pa.field('source_native_id',pa.string()),
    pa.field('source_event_id',pa.string()),
    pa.field('source_dataset_id',pa.string()),
    pa.field('source_occurrence_ids',pa.list_(pa.string())),
    pa.field('source_occurrence_count',pa.int32(),False),
    pa.field('ecotype_detail',pa.string(),False),
    pa.field('category',pa.string(),False),
    pa.field('classification_status',pa.string(),False),
    pa.field('native_classification',pa.string()),
    pa.field('source_description',pa.string()),
    pa.field('observed_at_raw',pa.string()),
    pa.field('observed_date_raw',pa.string()),
    pa.field('pod_group_assignments',pa.list_(pa.struct([
        pa.field('label',pa.string(),False),
        pa.field('group_type',pa.string(),False),
        pa.field('source_term',pa.string()),
        pa.field('evidence_field',pa.string(),False),
        pa.field('evidence_text',pa.string()),
        pa.field('confidence',pa.string(),False),
        pa.field('conflicting',pa.bool_(),False),
    ])),False),
    pa.field('pod_group_status',pa.string(),False),
    pa.field('classification_evidence_json',pa.string(),False),
    pa.field('observed_at_utc',pa.timestamp('us',tz='UTC')),
    pa.field('time_start_utc',pa.timestamp('us',tz='UTC')),
    pa.field('time_end_utc',pa.timestamp('us',tz='UTC')),
    pa.field('time_precision',pa.string(),False),
    pa.field('native_date',pa.date32()),
    pa.field('source_timezone',pa.string(),False),
    pa.field('utc_date',pa.date32()),
    pa.field('latitude',pa.float64()),
    pa.field('longitude',pa.float64()),
    pa.field('latitude_raw',pa.string()),
    pa.field('longitude_raw',pa.string()),
    pa.field('coordinate_uncertainty_m',pa.float64()),
    pa.field('coordinate_transform',pa.string(),False),
    pa.field('coordinate_role',pa.string()),
    pa.field('h3_r6',pa.string()),
    pa.field('h3_precision_status',pa.string(),False),
    pa.field('source_license',pa.string(),False),
    pa.field('source_use_class',pa.string(),False),
    pa.field('source_retrieved_at_utc',pa.timestamp('us',tz='UTC')),
    pa.field('qc_flags',pa.list_(pa.string()),False),
])


def reported_groups(evidence, detail):
    """Preserve explicit pod labels and typed native groups, never infer a pod.

    T-prefixed identifiers alone do not establish a pod or matriline; retain
    them as identifier_untyped for point detail, excluded from pod totals.
    Likewise an individual J35 does not create a J-pod assignment.
    """
    result={}
    for item in evidence:
        kind=item['ASSOCIATION_KIND'];label=item['ASSOCIATION_VALUE']
        text=item.get('EVIDENCE_TEXT') or ''
        term=None
        if kind=='POD' and item['RULE_ID']=='pod.explicit':
            group_type='pod';term='pod'
        elif kind=='SOCIAL_GROUP':
            group_type='identifier_untyped'
            quoted=re.escape(label)
            match=re.search(r'\b(matriline|group|pod)\s*(?:of\s*)?'+quoted+r'\b|\b'+quoted+r'\s*(matriline|group|pod)\b',text,re.IGNORECASE)
            if match:
                term=(match.group(1) or match.group(2)).lower()
                group_type={'pod':'source_named_pod','group':'source_named_group','matriline':'source_named_matriline'}[term]
        else:
            continue
        key=(label,group_type,item['EVIDENCE_FIELD'])
        result[key]=dict(label=label,group_type=group_type,source_term=term,
            evidence_field=item['EVIDENCE_FIELD'],evidence_text=text,
            confidence=item['CONFIDENCE'],conflicting=bool(item.get('CONFLICTING',False)))
    assignments=sorted(result.values(),key=lambda a:(a['group_type'],a['label'],a['evidence_field']))
    groups={(a['group_type'],a['label']) for a in assignments if a['group_type']!='identifier_untyped'}
    conflicting=any(a['conflicting'] for a in assignments)
    status='conflicting_source_evidence' if conflicting else 'multiple_reported' if len(groups)>1 else 'reported' if groups else 'unknown'
    return assignments,status


def prefer_source_record(previous, incoming):
    """Prefer live source evidence over copied historical CSV supplements.

    A current-feed collection may carry overlapping archive CSVs. Their new
    filesystem retrieval time does not make old archive values a live correction.
    For equally supported records, retain first retrieval for unchanged payloads
    and use the later retrieval for actual corrections. Never delete absent IDs.
    """
    if previous is None:
        return incoming
    if previous['SOURCE_RECORD_ID'] != incoming['SOURCE_RECORD_ID']:
        raise ValueError('upsert identity mismatch')
    if previous['SOURCE_PAYLOAD'] == incoming['SOURCE_PAYLOAD']:
        return previous
    def rank(row):
        payload=json.loads(row['SOURCE_PAYLOAD'])
        return ('_source_file' not in payload,pd.Timestamp(row['SOURCE_RETRIEVED_AT_UTC']))
    return incoming if rank(incoming)>=rank(previous) else previous


def time_support(row, settings, payload):
    """Return actual source support; never use the toolkit's synthetic noon.

    Naive clocks use the configured provider timezone. Date-only iNaturalist
    records use their native observed_time_zone; no UTC date is invented if it
    is absent. Ambiguous/nonexistent DST clocks remain unknown.
    """
    zone = settings.timezone
    if row['SOURCE']=='INATURALIST':
        zone = payload.get('observed_time_zone') or 'unknown'
    elif row['SOURCE']=='GBIF':
        # A Darwin Core calendar eventDate is not evidence of UTC. The generic
        # configured UTC parser default cannot establish a publisher timezone.
        # Only an offset-bearing source timestamp establishes UTC support here.
        zone = 'unknown'
    flags=[]
    raw=row.get('OBSERVED_AT_RAW')
    stamp=None
    if _contains_clock(raw):
        # Offset-aware timestamps need no assumed zone.
        has_offset=bool(re.search(r'(?:Z|[+-]\d{2}:?\d{2})$',str(raw)))
        if zone!='unknown' or has_offset:
            try:
                stamp=_parse_timestamp(raw,zone if zone!='unknown' else 'UTC')
            except (ValueError,ZoneInfoNotFoundError):
                stamp=None
    native_day=_parse_explicit_date(row.get('OBSERVED_DATE_RAW'),settings.date_formats)
    if stamp is not None:
        local_day=stamp.tz_convert(zone).date() if zone!='unknown' else None
        if native_day and local_day and native_day!=local_day:
            flags.append('NATIVE_DATE_TIMESTAMP_CONFLICT')
        precision='second' if re.search(r':\d{2}:\d{2}',str(raw)) else 'minute'
        # String precision is representation support, not clock accuracy.
        width=pd.Timedelta(seconds=1 if precision=='second' else 60)
        if re.search(r':\d{2}\.\d+',str(raw)):
            precision='subsecond'
            digits=len(re.search(r':\d{2}\.(\d+)',str(raw)).group(1))
            width=pd.Timedelta(microseconds=max(1,10**max(0,6-digits)))
        return dict(observed_at_utc=stamp,time_start_utc=stamp,time_end_utc=stamp+width,
                    time_precision=precision,native_date=native_day or local_day,
                    source_timezone=zone,utc_date=stamp.date()),flags
    native_day=native_day or _parse_explicit_date(raw,settings.date_formats)
    start=end=None
    if native_day and zone!='unknown':
        try:
            start=pd.Timestamp(native_day).tz_localize(zone,ambiguous='NaT',nonexistent='NaT')
            end=(pd.Timestamp(native_day)+pd.Timedelta(days=1)).tz_localize(zone,ambiguous='NaT',nonexistent='NaT')
            start,end=start.tz_convert('UTC'),end.tz_convert('UTC')
        except (ValueError,ZoneInfoNotFoundError):
            start=end=None
    if start is not None and (pd.isna(start) or pd.isna(end)):
        start=end=None
    if _contains_clock(raw):
        flags.append('UNKNOWN_SOURCE_TIMEZONE' if zone=='unknown' else 'UNPARSED_OR_AMBIGUOUS_SOURCE_CLOCK')
        # A failed clock may identify a day, but must not masquerade as a time.
    elif native_day and zone=='unknown':
        flags.append('UNKNOWN_SOURCE_TIMEZONE')
    native_precision=('subsecond' if re.search(r':\d{2}\.\d+',str(raw)) else
                      'second' if re.search(r':\d{2}:\d{2}',str(raw)) else 'minute') if _contains_clock(raw) else None
    return dict(observed_at_utc=None,time_start_utc=start,time_end_utc=end,
                time_precision=native_precision or ('day' if native_day else 'unknown'),native_date=native_day,
                source_timezone=zone,utc_date=(start.date() if start is not None and
                start.date()==(end-pd.Timedelta(microseconds=1)).date() else None)),flags


def period_start(day: date, grain: str):
    if grain=='daily': return day
    if grain=='weekly': return day-pd.Timedelta(days=day.weekday())
    if grain=='monthly': return day.replace(day=1)
    if grain=='yearly': return day.replace(month=1,day=1)
    raise ValueError(grain)


def containing_period(start, end, grain):
    """A coarse source interval belongs only to a bin containing all its support."""
    if start is None or end is None or pd.isna(start) or pd.isna(end): return None
    start=pd.Timestamp(start).tz_convert('UTC')
    end=pd.Timestamp(end).tz_convert('UTC')
    left=period_start(start.date(),grain)
    right=period_start((end-pd.Timedelta(microseconds=1)).date(),grain)
    return left if left==right else None


def build_study_sightings(source_rows, config, mask, registry, *, start, end):
    """One row per stable source record; retain overlap without false identity claims.

    Review all exclusion counts. No effort/absence surface or inferred whales is
    created. Outside-mask points are excluded even if their cell intersects it.
    """
    selected=[]; exclusions=[]; seen=set(); prepared=prep(mask)
    policy=config.observation_policy
    for row in source_rows:
        sid=row['SOURCE_RECORD_ID']
        if sid in seen: raise ValueError(f'duplicate source record: {sid}')
        seen.add(sid)
        if not policy.accepts(row):
            exclusions.append((sid,'NON_ORCA'));continue
        if row.get('SOURCE_QC_STATUS') not in (None,'ACCEPTED'):
            exclusions.append((sid,'SOURCE_QC_QUARANTINED'));continue
        payload=json.loads(row['SOURCE_PAYLOAD'])
        setting=config.collection.sources[row['SOURCE'].lower()]
        times,flags=time_support(row,setting,payload)
        if times['time_start_utc'] is None and times['native_date'] is not None and (
            times['native_date']<start.date() or times['native_date']>end.date()
        ):
            exclusions.append((sid,'OUTSIDE_NATIVE_DATE_WINDOW'));continue
        if times['time_end_utc'] is not None and (times['time_end_utc']<=start or times['time_start_utc']>=end):
            exclusions.append((sid,'OUTSIDE_REQUESTED_TIME'));continue
        coords=_coordinates(row,config)
        if coords is None:
            exclusions.append((sid,'INVALID_OR_OUTSIDE_ENVELOPE_COORDINATES'));continue
        lat,lon,transform=coords
        if not prepared.covers(Point(lon,lat)):
            exclusions.append((sid,'OUTSIDE_EXACT_STUDY_MASK'));continue
        # Inferred sign/swap corrections are recorded, never silent.
        if transform!='NONE':flags.append(transform)
        cell=h3.latlng_to_cell(lat,lon,6)
        if cell not in registry:
            raise ValueError(f'approved mask point outside approved R6 registry: {sid}')
        evidence=policy.evidence(row);detail=policy.classify(evidence)
        assignments,group_status=reported_groups(evidence,detail)
        if detail=='MIXED':flags.append('CONFLICTING_ECOTYPE_EVIDENCE')
        category={'SRKW':'srkw','TRANSIENT':'transient'}.get(detail,'other')
        uncertainty=row.get('COORDINATE_UNCERTAINTY_M')
        if uncertainty is not None and (not math.isfinite(uncertainty) or uncertainty<0):
            flags.append('INVALID_COORDINATE_UNCERTAINTY');uncertainty=None
        precision='assignment_only_unknown_accuracy' if uncertainty is None else 'assignment_only_reported_uncertainty'
        if uncertainty is None:flags.append('UNKNOWN_COORDINATE_UNCERTAINTY')
        if times['time_start_utc'] is None:flags.append('UNBINNED_TIME')
        if row['SOURCE']=='ACARTIA':flags.append('CREATED_FIELD_AS_EVENT_TIME_PER_SOURCE_POLICY')
        role=payload.get('coordinate_role')
        if role and 'UNSPECIFIED' in role:flags.append('COORDINATE_ROLE_UNSPECIFIED')
        selected.append(dict(event_id='orca:source:v1:'+hashlib.sha256(sid.encode()).hexdigest()[:24],
            source=row['SOURCE'],source_record_id=sid,source_native_id=row.get('SOURCE_NATIVE_ID'),
            source_event_id=row.get('SOURCE_EVENT_ID'),source_dataset_id=row.get('SOURCE_DATASET_ID'),
            source_occurrence_ids=row.get('SOURCE_OCCURRENCE_IDS'),
            source_occurrence_count=row.get('SOURCE_OCCURRENCE_COUNT') or 1,
            ecotype_detail=detail,category=category,
            classification_status='unknown' if detail=='UNKNOWN' else 'mixed' if detail=='MIXED' else 'source_supported',
            native_classification=row.get('POD_ECOTYPE_RAW'),source_description=row.get('DESCRIPTION_RAW'),
            observed_at_raw=row.get('OBSERVED_AT_RAW'),observed_date_raw=row.get('OBSERVED_DATE_RAW'),
            pod_group_assignments=assignments,pod_group_status=group_status,
            classification_evidence_json=json.dumps(evidence,sort_keys=True),
            **times,latitude=lat,longitude=lon,latitude_raw=row.get('LATITUDE_RAW'),longitude_raw=row.get('LONGITUDE_RAW'),
            coordinate_uncertainty_m=uncertainty,coordinate_transform=transform,coordinate_role=role,h3_r6=cell,
            h3_precision_status=precision,source_license=row.get('SOURCE_LICENSE') or 'UNKNOWN',
            source_use_class=row.get('SOURCE_USE_CLASS') or 'INTERNAL_ONLY',
            source_retrieved_at_utc=row.get('SOURCE_RETRIEVED_AT_UTC'),qc_flags=sorted(set(flags))))
    selected.sort(key=lambda r:r['event_id'])
    return pa.Table.from_pylist(selected,schema=SIGHTINGS_SCHEMA),exclusions
