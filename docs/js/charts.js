async function loadPantheonData() {
  const summary = document.getElementById("summary");
  const table = document.getElementById("results-table");
  if (!summary || !table) return;

  let payload = { results: [] };
  try {
    const response = await fetch("../assets/web_data.json");
    payload = await response.json();
  } catch (error) {
    summary.textContent = `Unable to load benchmark data: ${error}`;
    return;
  }

  const rows = payload.results || [];
  summary.textContent = `${rows.length} benchmark rows loaded.`;
  if (!rows.length) {
    table.innerHTML = "<tbody><tr><td>No benchmark data generated yet.</td></tr></tbody>";
    return;
  }

  const columns = ["Report Timestamp", "GPU Name", "Test Name", "Score", "Unit", "Avg Power (W)", "Max Temp (C)"];
  table.innerHTML = [
    `<thead><tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr></thead>`,
    `<tbody>${rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`,
  ].join("");

  const chartEl = document.getElementById("chart");
  if (chartEl && window.ApexCharts) {
    const recent = rows.slice(-20);
    new ApexCharts(chartEl, {
      chart: { type: "bar", height: 320 },
      series: [{ name: "Score", data: recent.map((row) => Number(row.Score) || 0) }],
      xaxis: { categories: recent.map((row) => `${row["GPU Name"]} ${row["Test Name"]}`) },
    }).render();
  }
}

loadPantheonData();
