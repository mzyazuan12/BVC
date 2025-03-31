import csv
import random
import datetime

# january baselines: from 2014..2023 
jan_baselines = {
    2014: 721.75,
    2015: 678.24,
    2016: 437.24,
    2017: 457.42,
    2018: 435.99,
    2019: 408.39,
    2020: 411.88,
    2021: 439.42,
    2022: 496.07,
    2023: 669.70,
}

input_csv = "s12.csv"
output_csv = "s.csv"

# how big can the monthly random step be (±units in 'Gross_Revenue')
STEP_RANGE = 15.0

# how strongly to pull each month toward the "annual" slope. smaller means weaker pull
DRIFT_STRENGTH = 0.2

def get_jan_baseline(year):
   
    return jan_baselines.get(year)  

def annual_interpolation(date_obj):
    year = date_obj.year
    month = date_obj.month  # 1..12
    
    this_year_val = get_jan_baseline(year)
    next_year_val = get_jan_baseline(year + 1)
    
    # fraction from january=0 to dec=11 
    fraction = (month - 1) / 12.0
    desired = this_year_val + fraction * (next_year_val - this_year_val)
    return desired

# read CSV
with open(input_csv, newline='', encoding='utf-8') as infile:
    import_csv = list(csv.reader(infile))
    header = import_csv[0]
    rows = import_csv[1:]

# find index of "Gross_Revenue"
try:
    rev_idx = header.index("Gross_Revenue")
except ValueError:
    raise Exception("Could not find 'Gross_Revenue' in CSV header")

# parse and sort all rows by date
import datetime
def parse_date(r):
    return datetime.datetime.strptime(r[0], "%Y-%m-%d")

rows.sort(key=parse_date)

randomized_rows = []
prev_val = None

for row in rows:
    date_str = row[0]
    date_obj = parse_date(row)
    # get the "desired" baseline for this exact month
    desired_val = annual_interpolation(date_obj)
    if prev_val is None:
        # first row -> just set it near desired
        current_val = desired_val
    else:
        # random step from prev_val
        step = random.uniform(-STEP_RANGE, STEP_RANGE)
        # naive new_val = prev_val + step
        naive_val = prev_val + step
        # also drift some portion of the difference to desired
        gap = desired_val - naive_val
        drift = DRIFT_STRENGTH * gap
        current_val = naive_val + drift
    # store
    row[rev_idx] = f"{current_val:.2f}"
    randomized_rows.append(row)
    prev_val = current_val
with open(output_csv, "w", newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(randomized_rows)

