# Data requirements

Curriculum Aula numbers and physical OneLake folder names are independent. The
notebooks keep their current curriculum numbering, while data continues to use
the folder belonging to the lesson from which the notebook originated.

## Curriculum-to-physical-root map

| Current curriculum Aula | Physical OneLake root(s) used |
| --- | --- |
| DMML Aula 0 | `DMML_Aula0` |
| DMML Aula 1 | `DMML_Aula1` |
| DMML Aula 2 | `DMML_Aula2` for the original Aula 2 material; `DMML_Aula3` for Partes 7–13 (old Aula 3) |
| DMML Aula 3 | `DMML_Aula4` for its data-bearing old-Aula-4 material |
| DMML Aula 4 | `DMML_Aula5` |
| DMML Aula 5 | `DMML_Aula6` |
| DMML Aula 6 | `DMML_Aula7` |
| DMML Aula 7 | `DMML_Aula8` |
| DMML Aula 8 | `AMLAI_Aula1` for the moved SVM exercise |
| AMLAI Aula 1 | No manual-upload root is referenced |
| AMLAI Aula 2 | `AMLAI_Aula1` for old-Aula-1 material; old-Aula-2 neural material retains `AMLAI_Aula2` semantics and uses generated churn assets |
| AMLAI Aula 3 | `AMLAI_Aula3` for old-Aula-3 material; `AMLAI_Aula4` for old-Aula-4 material |
| AMLAI Aula 4 | `AMLAI_Aula5` |
| AMLAI Aula 5 | `AMLAI_Aula6` |
| AMLAI Aula 6 | `AMLAI_Aula7` |
| AMLAI Aula 7 | Original named tables and public URLs; no renumbered upload root |
| AMLAI Aula 8 | No manual-upload root is referenced |

The churn notebook requires the case-sensitive path
`/lakehouse/default/Files/DMML_Aula4/ChurnData.csv`.

## Verified source pointers

- `DMML_Aula4/ChurnData.csv`: exact IBM Developer Skills Network course file:
  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv
  This is the 200-row, 28-column telecommunications churn dataset expected by
  the notebook, not the unrelated 7,043-row Telco Customer Churn dataset.
- `AMLAI_Aula1/pulsar_stars.csv`: notebook-compatible HTRU2 file with the
  expected header:
  https://raw.githubusercontent.com/alexandrehsd/predicting-pulsar-stars/master/pulsar_stars.csv
  The authoritative dataset record is available from UCI at
  https://archive.ics.uci.edu/dataset/372/htru2 (CC BY 4.0). The UCI data rows
  match, but its current CSV uses different column names, so it is not a
  drop-in replacement without renaming the headers.

## Manual uploads

The following are the unique concrete manual file paths extracted from notebook
read operations. Paths are grouped by their physical root.

### `AMLAI_Aula1`

- `/lakehouse/default/Files/AMLAI_Aula1/home-data-for-ml-course/test.csv`
- `/lakehouse/default/Files/AMLAI_Aula1/home-data-for-ml-course/train.csv`
- `/lakehouse/default/Files/AMLAI_Aula1/melb_data.csv`
- `/lakehouse/default/Files/AMLAI_Aula1/pulsar_stars.csv`

### `AMLAI_Aula3`

- `/lakehouse/default/Files/AMLAI_Aula3/concrete.csv`
- `/lakehouse/default/Files/AMLAI_Aula3/red-wine.csv`

### `AMLAI_Aula4`

- `/lakehouse/default/Files/AMLAI_Aula4/concrete.csv`
- `/lakehouse/default/Files/AMLAI_Aula4/fuel.csv`
- `/lakehouse/default/Files/AMLAI_Aula4/hotel.csv`
- `/lakehouse/default/Files/AMLAI_Aula4/ion.csv`
- `/lakehouse/default/Files/AMLAI_Aula4/red-wine.csv`
- `/lakehouse/default/Files/AMLAI_Aula4/spotify.csv`

### `AMLAI_Aula5`

- `/lakehouse/default/Files/AMLAI_Aula5/computer-vision-resources/car_feature.jpg`
- `/lakehouse/default/Files/AMLAI_Aula5/computer-vision-resources/car_illus.jpg`
- `/lakehouse/default/Files/AMLAI_Aula5/computer-vision-resources/k.jpg`
- `/lakehouse/default/Files/AMLAI_Aula5/computer-vision-resources/machinelearning.csv`

These notebooks also read directory resources at
`AMLAI_Aula5/car-or-truck/train`, `AMLAI_Aula5/car-or-truck/valid`,
`AMLAI_Aula5/cv-course-models/cv-course-models/inceptionv1`, and
`AMLAI_Aula5/cv-course-models/cv-course-models/vgg16-pretrained-base`.

### `AMLAI_Aula6`

- `/lakehouse/default/Files/AMLAI_Aula6/Google_Stock_Price_Test.csv`
- `/lakehouse/default/Files/AMLAI_Aula6/Google_Stock_Price_Train.csv`
- `/lakehouse/default/Files/AMLAI_Aula6/international-airline-passengers.csv`
- `/lakehouse/default/Files/AMLAI_Aula6/restaurant.json`
- `/lakehouse/default/Files/AMLAI_Aula6/review_vectors.npy`
- `/lakehouse/default/Files/AMLAI_Aula6/spam.csv`
- `/lakehouse/default/Files/AMLAI_Aula6/tsla.us.txt`
- `/lakehouse/default/Files/AMLAI_Aula6/yelp_ratings.csv`

One text-generation tutorial also scans CSV files placed directly in the
`/lakehouse/default/Files/AMLAI_Aula6/` directory.

### `AMLAI_Aula7`

- `/lakehouse/default/Files/AMLAI_Aula7/ar.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/book_sales.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/candy_production.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/dataset.txt`
- `/lakehouse/default/Files/AMLAI_Aula7/flu-trends.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/reserve.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/store-sales-time-series-forecasting/holidays_events.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/store-sales-time-series-forecasting/test.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/store-sales-time-series-forecasting/train.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/tunnel.csv`
- `/lakehouse/default/Files/AMLAI_Aula7/us-retail-sales.csv`

### `DMML_Aula0`

- `/lakehouse/default/Files/DMML_Aula0/powerlifting-database/meets.csv`
- `/lakehouse/default/Files/DMML_Aula0/powerlifting-database/openpowerlifting.csv`
- `/lakehouse/default/Files/DMML_Aula0/top-things-redit/gaming.csv`
- `/lakehouse/default/Files/DMML_Aula0/top-things-redit/movies.csv`
- `/lakehouse/default/Files/DMML_Aula0/wine-reviews/winemag-data-130k-v2.csv`
- `/lakehouse/default/Files/DMML_Aula0/wine-reviews/winemag-data_first150k.csv`
- `/lakehouse/default/Files/DMML_Aula0/youtube-new/CAvideos.csv`
- `/lakehouse/default/Files/DMML_Aula0/youtube-new/GBvideos.csv`

### `DMML_Aula1`

- `/lakehouse/default/Files/DMML_Aula1/home-data-for-ml-course/train.csv`
- `/lakehouse/default/Files/DMML_Aula1/melbourne-housing-snapshot/melb_data.csv`

### `DMML_Aula2`

- `/lakehouse/default/Files/DMML_Aula2/cancer.csv`
- `/lakehouse/default/Files/DMML_Aula2/candy.csv`
- `/lakehouse/default/Files/DMML_Aula2/fifa.csv`
- `/lakehouse/default/Files/DMML_Aula2/flight_delays.csv`
- `/lakehouse/default/Files/DMML_Aula2/ign_scores.csv`
- `/lakehouse/default/Files/DMML_Aula2/insurance.csv`
- `/lakehouse/default/Files/DMML_Aula2/iris.csv`
- `/lakehouse/default/Files/DMML_Aula2/museum_visitors.csv`
- `/lakehouse/default/Files/DMML_Aula2/spotify.csv`

### `DMML_Aula3`

- `/lakehouse/default/Files/DMML_Aula3/Building_Permits.csv`
- `/lakehouse/default/Files/DMML_Aula3/NFL Play by Play 2009-2017 (v4).csv`
- `/lakehouse/default/Files/DMML_Aula3/PoliceKillingsUS.csv`
- `/lakehouse/default/Files/DMML_Aula3/catalog.csv`
- `/lakehouse/default/Files/DMML_Aula3/database.csv`
- `/lakehouse/default/Files/DMML_Aula3/home-data-for-ml-course/test.csv`
- `/lakehouse/default/Files/DMML_Aula3/home-data-for-ml-course/train.csv`
- `/lakehouse/default/Files/DMML_Aula3/ks-projects-201612.csv`
- `/lakehouse/default/Files/DMML_Aula3/ks-projects-201801.csv`
- `/lakehouse/default/Files/DMML_Aula3/melb_data.csv`
- `/lakehouse/default/Files/DMML_Aula3/pakistan_intellectual_capital.csv`

### `DMML_Aula4`

- `/lakehouse/default/Files/DMML_Aula4/ChurnData.csv`
- `/lakehouse/default/Files/DMML_Aula4/advertising.csv`
- `/lakehouse/default/Files/DMML_Aula4/concrete.csv`
- `/lakehouse/default/Files/DMML_Aula4/creditcard.csv`
- `/lakehouse/default/Files/DMML_Aula4/fe-course-data/accidents.csv`
- `/lakehouse/default/Files/DMML_Aula4/fe-course-data/ames.csv`
- `/lakehouse/default/Files/DMML_Aula4/fe-course-data/autos.csv`
- `/lakehouse/default/Files/DMML_Aula4/fe-course-data/concrete.csv`
- `/lakehouse/default/Files/DMML_Aula4/fe-course-data/customer.csv`
- `/lakehouse/default/Files/DMML_Aula4/feature-engineering-data/train_sample.csv`
- `/lakehouse/default/Files/DMML_Aula4/pima-indians-diabetes.data`

### `DMML_Aula5`

- `/lakehouse/default/Files/DMML_Aula5/AER_credit_card_data.csv`
- `/lakehouse/default/Files/DMML_Aula5/Ecommerce Customers.txt`
- `/lakehouse/default/Files/DMML_Aula5/Salary_Data.csv`
- `/lakehouse/default/Files/DMML_Aula5/USA_Housing.csv`
- `/lakehouse/default/Files/DMML_Aula5/avocado.csv`

### `DMML_Aula6`

- `/lakehouse/default/Files/DMML_Aula6/ames.csv`
- `/lakehouse/default/Files/DMML_Aula6/autos.csv`
- `/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/test.csv`
- `/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/train.csv`
- `/lakehouse/default/Files/DMML_Aula6/melb_data.csv`
- `/lakehouse/default/Files/DMML_Aula6/winequality-red.csv`

### `DMML_Aula7`

- `/lakehouse/default/Files/DMML_Aula7/car_evaluation.csv`
- `/lakehouse/default/Files/DMML_Aula7/home-data-for-ml-course/test.csv`
- `/lakehouse/default/Files/DMML_Aula7/home-data-for-ml-course/train.csv`
- `/lakehouse/default/Files/DMML_Aula7/melb_data.csv`

### `DMML_Aula8`

- `/lakehouse/default/Files/DMML_Aula8/ames.csv`
- `/lakehouse/default/Files/DMML_Aula8/autos.csv`
- `/lakehouse/default/Files/DMML_Aula8/housing.csv`

## Not manual uploads

- HTTP(S) datasets and GenAI Aula 7 source URLs remain public runtime inputs.
- `Tables/df_clean`, `Tables/df_test`, other named tables, and parameterized
  `Tables/{table_name}` references are generated or workspace-managed tables.
- `Files/churn`, `Files/churn/raw/churn.csv`, transformed CSVs, submissions,
  model outputs, and inline recommendation examples are generated during the
  exercises rather than separate upload requirements.
