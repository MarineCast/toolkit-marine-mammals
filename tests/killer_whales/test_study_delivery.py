from datetime import date
from types import SimpleNamespace
import pandas as pd
import pytest
import h3
from shapely.geometry import box
from marine_mammal_toolkit.cetaceans.killer_whales.observations.study_delivery import (
    time_support, containing_period, build_study_sightings, prefer_source_record, reported_groups,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import load_sightings_config
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from pathlib import Path


def test_native_day_is_not_assigned_fake_utc_day():
    setting=SimpleNamespace(timezone='America/Los_Angeles',date_formats=('%Y-%m-%d',))
    result,_=time_support({'SOURCE':'TWM','OBSERVED_DATE_RAW':'2026-03-08','OBSERVED_AT_RAW':None},setting,{})
    assert result['time_end_utc']-result['time_start_utc']==pd.Timedelta(hours=23)
    assert result['observed_at_utc'] is None and result['utc_date'] is None
    assert containing_period(result['time_start_utc'],result['time_end_utc'],'daily') is None
    assert containing_period(result['time_start_utc'],result['time_end_utc'],'monthly')==date(2026,3,1)
    assert containing_period(result['time_start_utc'],result['time_end_utc'],'weekly') is None


def test_offset_timestamp_uses_actual_utc_not_native_date_or_noon():
    setting=SimpleNamespace(timezone='UTC',date_formats=('%Y-%m-%d',))
    result,flags=time_support({'SOURCE':'INATURALIST','OBSERVED_DATE_RAW':'2026-01-01','OBSERVED_AT_RAW':'2026-01-01T23:59:00-08:00'},setting,{'observed_time_zone':'America/Los_Angeles'})
    assert result['utc_date']==date(2026,1,2)
    assert result['time_start_utc']==pd.Timestamp('2026-01-02T07:59:00Z')
    assert not flags


def test_unknown_inaturalist_zone_does_not_assume_utc_day():
    setting=SimpleNamespace(timezone='UTC',date_formats=('%Y-%m-%d',))
    result,_=time_support({'SOURCE':'INATURALIST','OBSERVED_DATE_RAW':'2026-01-01','OBSERVED_AT_RAW':None},setting,{})
    assert result['time_start_utc'] is None and result['native_date']==date(2026,1,1)


def test_gbif_date_only_does_not_acquire_utc_from_parser_default():
    setting=SimpleNamespace(timezone='UTC',date_formats=('%Y-%m-%d',))
    result,_=time_support({'SOURCE':'GBIF','OBSERVED_DATE_RAW':'2026-01-01','OBSERVED_AT_RAW':None},setting,{})
    assert result['native_date']==date(2026,1,1)
    assert result['source_timezone']=='unknown' and result['time_start_utc'] is None
    precise,_=time_support({'SOURCE':'GBIF','OBSERVED_DATE_RAW':'2026-01-01','OBSERVED_AT_RAW':'2026-01-01T23:00:00Z'},setting,{})
    assert precise['observed_at_utc']==pd.Timestamp('2026-01-01T23:00:00Z')
    native,flags=time_support({'SOURCE':'GBIF','OBSERVED_DATE_RAW':'2026-01-01','OBSERVED_AT_RAW':'2026-01-01T23:00:00'},setting,{})
    assert native['time_precision']=='second' and native['time_start_utc'] is None
    assert 'UNKNOWN_SOURCE_TIMEZONE' in flags


def test_pods_are_not_created_from_member_ids_or_untyped_t_identifiers():
    def e(kind,value,text,rule):
        return dict(ASSOCIATION_KIND=kind,ASSOCIATION_VALUE=value,EVIDENCE_TEXT=text,RULE_ID=rule,EVIDENCE_FIELD='DESCRIPTION_RAW',CONFIDENCE='EXPLICIT',CONFLICTING=False)
    rows=[e('POD','J','J35','pod.from_member'),e('SOCIAL_GROUP','T46','J35 with T46','group.transient_id')]
    groups,status=reported_groups(rows,'MIXED')
    assert status=='unknown' and len(groups)==1 and groups[0]['group_type']=='identifier_untyped'
    rows=[e('POD','J','J pod and K pod','pod.explicit'),e('POD','K','J pod and K pod','pod.explicit'),e('SOCIAL_GROUP','T46','T46 matriline','group.transient_id')]
    groups,status=reported_groups(rows,'MIXED')
    assert status=='multiple_reported'
    assert {(g['label'],g['group_type']) for g in groups}=={('J','pod'),('K','pod'),('T46','source_named_matriline')}


def test_containment_monday_and_year_boundary():
    assert containing_period(pd.Timestamp('2025-12-31T23:00Z'),pd.Timestamp('2026-01-01T01:00Z'),'yearly') is None
    assert containing_period(pd.Timestamp('2026-01-04T23:00Z'),pd.Timestamp('2026-01-05T01:00Z'),'weekly') is None
    assert containing_period(pd.Timestamp('2026-01-05T01:00Z'),pd.Timestamp('2026-01-05T01:01Z'),'weekly')==date(2026,1,5)


def test_current_live_correction_beats_later_copied_old_archive():
    live=dict(SOURCE_RECORD_ID='ACARTIA:1',SOURCE_PAYLOAD='{"value": 2}',SOURCE_RETRIEVED_AT_UTC=pd.Timestamp('2026-09-20T00:00Z'))
    archive=dict(SOURCE_RECORD_ID='ACARTIA:1',SOURCE_PAYLOAD='{"_source_file":"old.csv","value":1}',SOURCE_RETRIEVED_AT_UTC=pd.Timestamp('2026-10-16T00:00Z'))
    assert prefer_source_record(live,archive) is live
    assert prefer_source_record(archive,live) is live
    repeat=dict(live,SOURCE_RETRIEVED_AT_UTC=pd.Timestamp('2026-10-16T00:00Z'))
    assert prefer_source_record(live,repeat) is live
    correction=dict(repeat,SOURCE_PAYLOAD='{"value": 3}')
    assert prefer_source_record(live,correction) is correction


def test_same_place_time_reports_are_not_falsely_merged():
    _,config=load_sightings_config(config_path(),workspace_root=Path.cwd())
    def row(source,native,classification):
        return dict(SOURCE=source,SOURCE_RECORD_ID=source+':'+native,SOURCE_NATIVE_ID=native,SOURCE_PAYLOAD='{}',
                    SPECIES_RAW='Orcinus orca',POD_ECOTYPE_RAW=classification,DESCRIPTION_RAW=None,
                    OBSERVED_DATE_RAW='2026-01-01',OBSERVED_AT_RAW='2026-01-01T12:00:00Z',
                    LATITUDE_RAW='48.5',LONGITUDE_RAW='-123.0',SOURCE_QC_STATUS='ACCEPTED')
    rows=[row('TWM','1',''),row('ACARTIA','2','Biggs')]
    cell=h3.latlng_to_cell(48.5,-123,6)
    table,exclusions=build_study_sightings(rows,config,box(-124,48,-122,49),{cell},start=pd.Timestamp('2009-01-01T00:00:00Z'),end=pd.Timestamp('2027-01-01T00:00:00Z'))
    assert table.num_rows==2 and not exclusions
    result=table.to_pylist()
    assert len({r['event_id'] for r in result})==2
    assert {r['category'] for r in result}=={'other','transient'}
    unknown=next(r for r in result if r['category']=='other')
    assert unknown['classification_status']=='unknown'
    assert unknown['h3_r6']==cell
    assert unknown['pod_group_assignments']==[] and unknown['pod_group_status']=='unknown'
    assert unknown['source_description'] is None
    assert unknown['h3_precision_status']=='assignment_only_unknown_accuracy'
    with pytest.raises(ValueError,match='duplicate source record'):
        build_study_sightings([rows[0],rows[0]],config,box(-124,48,-122,49),{cell},start=pd.Timestamp('2009-01-01T00:00:00Z'),end=pd.Timestamp('2027-01-01T00:00:00Z'))
    old=row('GBIF','old','')
    old.update(OBSERVED_DATE_RAW='2000-01-01',OBSERVED_AT_RAW=None)
    table,excluded=build_study_sightings(rows+[old],config,box(-124,48,-122,49),{cell},start=pd.Timestamp('2009-01-01T00:00:00Z'),end=pd.Timestamp('2027-01-01T00:00:00Z'))
    assert table.num_rows==2 and excluded==[('GBIF:old','OUTSIDE_NATIVE_DATE_WINDOW')]
