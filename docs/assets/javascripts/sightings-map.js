/* Documentation-only synthetic record explorer. Leaflet is vendored locally. */
(async function () {
  "use strict";
  const root = document.querySelector("[data-sightings-demo]");
  if (!root) return;
  const summary = root.querySelector("[data-summary]");
  const status = root.querySelector("[data-map-status]");
  const labelFilter = root.querySelector("[data-label-filter]");
  const dateFilter = root.querySelector("[data-date-filter]");
  const list = root.querySelector("[data-record-list]");
  const styles = {
    SRKW: { className: "srkw", symbol: "S" },
    Transient: { className: "transient", symbol: "T" },
    Unresolved: { className: "unresolved", symbol: "?" },
  };
  const formatDate = value => new Intl.DateTimeFormat("en", {
    day: "numeric", month: "short", timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
  const node = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  try {
    if (!window.L) throw new Error("Map library unavailable");
    const response = await fetch(new URL(root.dataset.source, document.baseURI));
    if (!response.ok) throw new Error("Example fixture unavailable");
    const fixture = await response.json();
    if (fixture.synthetic !== true || !Array.isArray(fixture.records) ||
        fixture.records.some(record => record.synthetic !== true || !styles[record.label] ||
          !Number.isFinite(record.latitude) || !Number.isFinite(record.longitude))) {
      throw new Error("Expected a synthetic documentation fixture");
    }
    const records = fixture.records;
    const map = L.map(root.querySelector("[data-map]"), {
      scrollWheelZoom: false, minZoom: 7, maxZoom: 14,
    });
    const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
    }).addTo(map);
    tiles.on("tileerror", () => {
      status.hidden = false;
      status.textContent = "Some basemap tiles are unavailable. Example records and filters still work; reload when network access is available.";
    });
    const markers = L.layerGroup().addTo(map);
    const allBounds = L.latLngBounds(records.map(record => [record.latitude, record.longitude]));
    let selectedButton = null;
    const selectRecord = (marker, button) => {
      if (selectedButton) selectedButton.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-pressed", "true");
      selectedButton = button;
      marker.openPopup();
    };
    const popup = record => {
      const container = node("div", "sightings-popup");
      container.append(node("span", "sightings-popup__badge", "SYNTHETIC EXAMPLE"));
      container.append(node("strong", "sightings-popup__title", record.location));
      container.append(node("p", "", `${record.id} · ${formatDate(record.date)} 2025`));
      const details = node("dl", "sightings-popup__details");
      for (const [label, value] of [
        ["Example label", record.label],
        ["Latitude", `${record.latitude.toFixed(3)}°`],
        ["Longitude", `${record.longitude.toFixed(3)}°`],
        ["Provenance", "Fictional documentation fixture"],
      ]) {
        details.append(node("dt", "", label), node("dd", "", value));
      }
      container.append(details);
      return container;
    };
    const fit = visible => {
      const bounds = visible.length ? L.latLngBounds(visible.map(record => [record.latitude, record.longitude])) : allBounds;
      map.fitBounds(bounds, { padding: [28, 28], maxZoom: 11, animate: false });
    };
    function render() {
      markers.clearLayers();
      list.replaceChildren();
      selectedButton = null;
      const visible = records.filter(record =>
        (labelFilter.value === "all" || record.label === labelFilter.value) &&
        (dateFilter.value === "all" ||
          (dateFilter.value === "early" ? record.date <= "2025-06-04" : record.date >= "2025-06-05")));
      summary.textContent = `${visible.length} of ${records.length} synthetic example records · ${labelFilter.options[labelFilter.selectedIndex].text} · ${dateFilter.options[dateFilter.selectedIndex].text} 2025`;
      for (const record of visible) {
        const style = styles[record.label];
        const icon = L.divIcon({
          className: "sightings-marker",
          html: `<span class="sightings-symbol sightings-symbol--${style.className}">${style.symbol}</span>`,
          iconSize: [30, 30], iconAnchor: [15, 15], popupAnchor: [0, -14],
        });
        const marker = L.marker([record.latitude, record.longitude], {
          icon, keyboard: true,
          title: `${record.id}: ${record.label}, ${record.location}, synthetic example`,
          alt: `${record.id}: synthetic ${record.label} example record`,
        }).bindPopup(popup(record), { maxWidth: 270 }).addTo(markers);
        const button = node("button", "sightings-record");
        button.type = "button";
        button.dataset.recordId = record.id;
        button.setAttribute("aria-pressed", "false");
        button.setAttribute("aria-label", `Inspect ${record.id}, ${record.label}, ${record.location}, ${formatDate(record.date)}`);
        const symbol = node("span", `sightings-symbol sightings-symbol--${style.className}`, style.symbol);
        symbol.setAttribute("aria-hidden", "true");
        const text = node("span", "sightings-record__text");
        text.append(node("strong", "", record.location));
        text.append(node("span", "", `${record.id} · ${formatDate(record.date)} · ${record.label}`));
        button.append(symbol, text);
        button.addEventListener("click", () => {
          map.panTo(marker.getLatLng());
          selectRecord(marker, button);
        });
        marker.on("click", () => selectRecord(marker, button));
        marker.on("popupclose", () => {
          button.setAttribute("aria-pressed", "false");
          if (selectedButton === button) selectedButton = null;
        });
        list.append(button);
      }
      if (!visible.length) list.append(node("p", "sightings-explorer__empty", "No example records match these filters. This does not indicate absence of whales."));
      fit(visible);
    }
    labelFilter.addEventListener("change", render);
    dateFilter.addEventListener("change", render);
    root.querySelector("[data-reset]").addEventListener("click", () => {
      labelFilter.value = "all";
      dateFilter.value = "all";
      render();
    });
    render();
    if (window.ResizeObserver) new ResizeObserver(() => map.invalidateSize()).observe(root.querySelector("[data-map]"));
  } catch (error) {
    summary.textContent = "The interactive example could not load. Open the JSON fixture below to inspect the synthetic records.";
    labelFilter.disabled = true;
    dateFilter.disabled = true;
    root.querySelector("[data-reset]").disabled = true;
    status.hidden = false;
    status.textContent = "No live observations are fetched by this example.";
    console.error("Sightings documentation example:", error);
  }
})();
