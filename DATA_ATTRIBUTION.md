# Data sources, licenses and attribution

The processed data files in this repository (`phase1/data/`, `phase2/data/`) combine three upstream sources. They are released under **Creative Commons Attribution 4.0 (CC BY 4.0)**, because that is the most restrictive upstream license. If you reuse them, keep the credits below.

Changes made to all sources: time-zone conversion to Asia/Ho_Chi_Minh, a −1 h label shift for PM2.5, removal/flagging of invalid values, interpolation of gaps ≤ 3 h, daily aggregation, spatial aggregation of fire points, and derived variables (clusters, rules, anomaly scores).

## 1. PM2.5 — U.S. Embassy Hanoi monitor (U.S. Department of State AirNow), via OpenAQ

- Origin: U.S. Department of State Air Quality Monitoring Program (AirNow DOS), monitor at the U.S. Embassy, Hanoi (21.0219 N, 105.8188 E).
- Retrieved from: OpenAQ open data archive on AWS (`s3://openaq-data-archive`, locations 2539 and 7441), Nov 2026. The original feed (dosairnowdata.org) has been offline since March 2025.
- License: works of the U.S. Government are not subject to copyright in the U.S.; OpenAQ tags AirNow data as **"US Public Domain"**. OpenAQ asks users to acknowledge the original source; the OpenAQ archive license "varies by data provider".
- Disclaimer: embassy data are preliminary, near-real-time monitoring data and were not fully validated.
- Credit: *"PM2.5 data: U.S. Department of State Air Quality Monitoring Program (U.S. Embassy Hanoi), accessed via OpenAQ (openaq.org)."*
- Links: https://docs.openaq.org/about/terms · https://docs.openaq.org/docs/data-policy · https://registry.opendata.aws/openaq/

## 2. Weather — Open-Meteo Historical Weather API (ERA5 / ERA5-Land)

- License: **CC BY 4.0**. Open-Meteo requires a credit with a link and an indication of changes.
- Credit: *"Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0)."*
- Underlying reanalysis: ERA5, Copernicus Climate Change Service (C3S), CC BY 4.0. Credit: *"Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."*
- Citations:
  - Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. https://doi.org/10.5281/zenodo.7970649
  - Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present. C3S Climate Data Store. https://doi.org/10.24381/cds.adbb2d47
  - Hersbach, H. et al. (2020). The ERA5 global reanalysis. QJRMS 146, 1999–2049. https://doi.org/10.1002/qj.3803
- Links: https://open-meteo.com/en/licence · https://open-meteo.com/en/docs/historical-weather-api

## 3. Fires — NASA FIRMS (VIIRS S-NPP 375 m, MODIS 1 km)

- Files: FIRMS yearly country files for Vietnam (`firms.modaps.eosdis.nasa.gov/data/country/...`).
- License: NASA Earth science data are openly shared **without restriction** (EOSDIS Data Use Policy); citation is strongly requested, and data passed on to third parties should keep the citation and a link to the disclaimer.
- Credit: *"We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://www.earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS)."*
- Product DOIs (please verify on the FIRMS site for the exact collection used):
  - VIIRS S-NPP 375 m: https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002
  - MODIS Collection 6.1: https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.0061
- Disclaimer: data are provided "as is"; users bear all responsibility for their use. https://www.earthdata.nasa.gov/data/projects/lance/disclaimer
- Links: https://www.earthdata.nasa.gov/learn/use-data/data-use-policy

## 4. Calendar

Tết dates computed with the `lunardate` Python package; Hanoi lockdown dates (Directive 16) from Vietnamese news sources cited inside `phase1/02_bien_bo_sung.ipynb`.
