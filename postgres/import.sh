#!/bin/bash
set -e  # Exit on error

# echo "Waiting for PostgreSQL to be ready..."
# until pg_isready -h postgres -p 5432 -U airflow; do
#   sleep 2
# done

echo "PostgreSQL is ready. Starting import."

# Define database credentials
DB_NAME="airflow"
DB_USER="airflow"

# Define the table and CSV file
TABLE_NAME="bank_churners"
CSV_FILE="/docker-entrypoint-initdb.d/postgres_bankchurners.csv"

# Check if the table exists
TABLE_EXISTS=$(psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT to_regclass('$TABLE_NAME');")

if [[ "$TABLE_EXISTS" != "$TABLE_NAME" ]]; then
  echo "Table $TABLE_NAME does not exist. Creating table"
  psql -U "$DB_USER" -d "$DB_NAME" -a -f /docker-entrypoint-initdb.d/init.sql
else
  echo "Table $TABLE_NAME exists. Truncating data..."
  psql -U "$DB_USER" -d "$DB_NAME" -c "TRUNCATE TABLE $TABLE_NAME RESTART IDENTITY;"
fi

# psql -U "$DB_USER" -d "$DB_NAME" -a -f /docker-entrypoint-initdb.d/init.sql
# Run the import command
psql -U "$DB_USER" -d "$DB_NAME" -c "\copy $TABLE_NAME FROM '$CSV_FILE' WITH CSV HEADER;"

echo "CSV import completed!"