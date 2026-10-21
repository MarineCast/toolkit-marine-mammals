# Orca study delivery

`cetaceans.killer_whales.observations.study_delivery` provides a conservative
source-record adapter for an explicitly supplied, validated study water mask and
H3 R6 registry. It requires Shapely (available in the `imputation` extra), but
does not fit or load a model. The adapter accepts current canonical source records
and the reviewed sightings configuration. The caller owns source acquisition,
upserts, mask/registry hash verification, persistence and independent count QA.

One output row is one stable source record. `event_id` is a deterministic hash of
the complete namespaced `SOURCE_RECORD_ID`. Reviewed GBIF encounter grouping may
represent several occurrence rows; `source_occurrence_count` is a count of source
occurrence records, never a whale count. Separate reports at the same place/time
remain separate, including cross-source reports. The result is not a globally
deduplicated count of biological encounters. Do not substitute proximity-merged
composite IDs or equate observation-label imputation with source classification.

`SRKW` maps to `srkw`; `TRANSIENT` maps to `transient`. Reviewed explicit Bigg's
text and transient social-group IDs support `TRANSIENT` under the existing
interpretation rules. `NRKW`, `OFFSHORE`, `MIXED` and `UNKNOWN` map to display
category `other` while retaining their native detail, unknown/mixed status and
evidence. Unrecognized TWM abbreviations are not assigned an ecotype. `all` is a
selection union over `event_id`, not an additional stored category.

The adapter never uses the pipeline's synthetic noon as an observation time.
Offset-aware timestamps retain their actual UTC instant. Naive timestamps use
the configured provider timezone. iNaturalist date-only records use their native
`observed_time_zone`; absent timezone leaves UTC support unknown. Local days run
from local midnight to next local midnight, including DST.
GBIF date-only or offset-free source times retain an unknown timezone: the
generic UTC parser default is not publisher evidence. Offset-bearing source
timestamps can still establish UTC support.
Lexical timestamp precision is not a claim of clock accuracy. Actual event timestamps win for UTC
binning; conflicts with native dates remain flagged.

All temporal support is half-open. `containing_period` assigns a source interval
only if one complete UTC daily / Monday-start weekly / calendar monthly / yearly
bin contains it. Unknown UTC support can be counted only All time. Source-local
days crossing UTC midnight cannot silently enter UTC daily bins. An annual census
has a separate temporal grain and must never be replicated into daily sightings.

Coordinates are checked against the planning envelope and the **exact** approved
mask. Existing unambiguous source-coordinate correction rules record sign/swap
changes alongside raw coordinates. Unsupported/unlocated records are excluded
with explicit audit; their absence does not imply that all reports were located.
`h3_r6` is the exact coordinate-to-cell assignment with no nearest-water snapping.
Both `h3_precision_status` values describe indexing of reported coordinates only.
Unknown source uncertainty remains null, not zero; known uncertainty is retained
and does not imply the true whale position is confined to the assigned cell.

`qc_flags` and `source_occurrence_ids` are Arrow `list<string>` (the latter can
be null). Classification evidence is JSON text, not an Arrow nested struct.
Sparse aggregate count tables contain positive report counts only. Missing cells
or bins are unknown/no report, never surveyed zero or biological absence. The
registry establishes reporting geometry, not observer effort.

Sources and dates are opportunistic/provider-specific. A successful API response
can verify request pagination, not complete whale occurrence or survey coverage.
Retained history must survive current-only/partial feeds; deletions require
specific completeness evidence. Pin retrieval time, actual observed bounds,
configured source policy, source files and checksums, code revision, local diff,
study config/mask/registry checksums, exclusions, and schema/readback/count QA.

Keep exact source locations and generated products local under source rights.
UNKNOWN/INTERNAL_ONLY remains internal-only. In particular CWR publicly states
that derivative use requires written approval; acquisition alone does not grant
publication rights. Census rows from a missing original workbook remain retained
derivatives unless independently checked against primary evidence. Preserve
reported totals and component sums when they disagree.

Point detail retains `source_description`, `observed_at_raw`, `observed_date_raw`
and `pod_group_assignments`. Descriptions are source content; render plain text
or sanitize at the consumer boundary. Explicit J/K/L pod evidence can create pod
assignments; an individual ID such as J35 cannot. A T-prefixed native identifier
alone has `group_type=identifier_untyped`, not a pod/matriline. An explicit nearby
source term creates `source_named_group`, `source_named_pod`, or
`source_named_matriline`, retaining its evidence text. Unknown remains empty/unknown;
multiple reported labels remain multiple. Grid itemization counts distinct source
record IDs per typed label; exclude `identifier_untyped` from pod totals. Multi-pod
records can occur in several itemized rows, whose sum can exceed unique records.
