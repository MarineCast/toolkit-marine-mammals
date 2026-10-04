---
hide:
  - toc
---

# Example sightings map

Explore a small, fictional set of orca sighting records around the Salish Sea.
Filter the view, then select a marker or record to inspect its details.

<div class="sightings-explorer" data-sightings-demo data-source="../../../../assets/examples/orca-sightings-demo.json">
  <div class="sightings-explorer__heading">
    <div>
      <span class="sightings-explorer__eyebrow">ORCAS / SIGHTINGS</span>
      <h2>Salish Sea explorer</h2>
      <p>1–8 June 2025 · Fictional example records</p>
    </div>
    <span class="sightings-explorer__badge">Synthetic example</span>
  </div>
  <div class="sightings-explorer__controls">
    <label>Example label
      <select data-label-filter>
        <option value="all">All labels</option>
        <option value="SRKW">SRKW</option>
        <option value="Transient">Transient</option>
        <option value="Unresolved">Unresolved</option>
      </select>
    </label>
    <label>Example dates
      <select data-date-filter>
        <option value="all">1–8 June</option>
        <option value="early">1–4 June</option>
        <option value="late">5–8 June</option>
      </select>
    </label>
    <button type="button" class="sightings-explorer__reset" data-reset>Reset view</button>
  </div>
  <div class="sightings-explorer__summary" role="status" aria-live="polite" data-summary>Loading example records…</div>
  <div class="sightings-explorer__body">
    <div class="sightings-explorer__map-area">
      <div class="sightings-explorer__map" data-map aria-label="Interactive map of synthetic example sightings"></div>
      <div class="sightings-explorer__legend" aria-label="Example label legend">
        <span><i class="sightings-symbol sightings-symbol--srkw" aria-hidden="true">S</i> SRKW</span>
        <span><i class="sightings-symbol sightings-symbol--transient" aria-hidden="true">T</i> Transient</span>
        <span><i class="sightings-symbol sightings-symbol--unresolved" aria-hidden="true">?</i> Unresolved</span>
      </div>
    </div>
    <section class="sightings-explorer__records" aria-label="Example records">
      <div class="sightings-explorer__records-heading">
        <h3>Example records</h3>
        <span>Select to inspect</span>
      </div>
      <div class="sightings-explorer__list" data-record-list></div>
    </section>
  </div>
  <p class="sightings-explorer__network-status" data-map-status role="status" hidden></p>
  <p class="sightings-explorer__note">All points, dates, and labels are synthetic. These are example records, not whale counts, measured positions, or evidence of species distribution.</p>
</div>

<noscript>
  <p>This interactive example needs JavaScript. The linked JSON fixture below contains all twelve synthetic records and their provenance.</p>
</noscript>

## About this example

Each marker is one fictional record. SRKW, Transient, and Unresolved labels
illustrate how a record explorer can keep categories and missing labels distinct;
they are not provider identifications or imputation results. No observations were
collected and no model was fitted for this example.

The map and record list share [one inspectable JSON fixture](../../../assets/examples/orca-sightings-demo.json).
Basemap tiles require network access. Map data ©
[OpenStreetMap contributors](https://www.openstreetmap.org/copyright);
the map uses [Leaflet](https://leafletjs.com/).

## Work with your own observations

Use the [observation-query guide](../../../public-usage.md) to select providers,
dates, and geography, then inspect the returned audits and provenance. For a
regional product with an HTML report, follow [releases and reports](products.md).
Review source rights before publishing actual observation locations or derived maps.
