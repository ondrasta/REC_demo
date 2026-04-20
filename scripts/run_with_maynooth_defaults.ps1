# Optional: use group-project CSVs as app defaults when uploads are empty (edit paths if yours differ).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV = "c:\Users\ondra\OneDrive\Desktop\Maynooth university\Analytics Live\Group Project\Final data\Houly_Data_kWh.csv"
$env:REC_FEASIBILITY_DEFAULT_PV_CSV = "c:\Users\ondra\OneDrive\Desktop\Maynooth university\Analytics Live\Group Project\Final data\Timeseries_52.141_-10.269_SA3_1kWp_crystSi_14_35deg_0deg_2020_2020.csv"

if (-not (Test-Path -LiteralPath $env:REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV)) {
    Write-Error "Consumption CSV not found: $($env:REC_FEASIBILITY_DEFAULT_CONSUMPTION_CSV)"
}
if (-not (Test-Path -LiteralPath $env:REC_FEASIBILITY_DEFAULT_PV_CSV)) {
    Write-Error "PV CSV not found: $($env:REC_FEASIBILITY_DEFAULT_PV_CSV)"
}

streamlit run app.py
