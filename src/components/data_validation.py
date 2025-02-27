import pandas as pd
import os
import re

class DataValidator:

    def __init__(self, config, naming_pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$'):
        """
        Initializes the DataValidator with a CSV file.
        :param file_path: Path to the CSV file.
        :param naming_pattern: Regex pattern to validate column naming convention.
        """
        self.naming_pattern = naming_pattern
        self.data = None
        self.report = None
        self.config = config

    def load_data(self):
        """Loads CSV file into a Pandas DataFrame."""
        if not os.path.exists(self.config["data_ingestion"]["csv"]["preproc_data_path"]):
            raise FileNotFoundError(f'File not found: {self.config["data_ingestion"]["csv"]["preproc_data_path"]}')
        self.data = pd.read_csv(self.config["data_ingestion"]["csv"]["preproc_data_path"])

    def check_naming_convention(self):
        """Checks if column names follow the specified naming convention."""
        invalid_columns = [col for col in self.data.columns if not re.match(self.naming_pattern, col)]
        return invalid_columns
    
    def check_required_columns(self):
        """Checks if all required columns are present in the dataset."""
        missing_columns = [col for col in self.config["data_validation"]["required_columns"] if col not in self.data.columns]
        return missing_columns

    def analyze_metadata(self):
        """Analyzes metadata statistics including fill rate, sparsity, most frequent values, mean, median, and mode."""
        if self.data is None:
            raise ValueError("Data is not loaded. Run load_data() first.")

        report_data = []
        for column in self.data.columns:
            fill_rate = self.data[column].count() / len(self.data) * 100
            sparsity = 100 - fill_rate
            value_counts = self.data[column].value_counts()
            most_frequent = value_counts.nlargest(2).to_dict() if not value_counts.empty else {}
            mean = self.data[column].mean() if pd.api.types.is_numeric_dtype(self.data[column]) else None
            median = self.data[column].median() if pd.api.types.is_numeric_dtype(self.data[column]) else None
            mode = list(most_frequent.keys())[0] if most_frequent else None
            
            report_data.append([
                column, fill_rate, sparsity, most_frequent, mean, median, mode
            ])

        self.report = pd.DataFrame(report_data, columns=[
            'Column Name', 'Fill Rate (%)', 'Sparsity (%)', 'Most Frequent Values', 'Mean', 'Median', 'Mode'
        ])

    def save_report(self, output_path):
        """Saves the metadata report to a CSV file."""
        if self.report is None:
            raise ValueError("Report is not generated. Run analyze_metadata() first.")
        self.report.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")

    def run_validation(self):
        """Executes the full validation pipeline."""
        self.load_data()
        invalid_columns = self.check_naming_convention()
        if invalid_columns:
            print(f"Warning: Invalid column names detected - {invalid_columns}")
            return False
        
        missing_columns = self.check_required_columns()
        if missing_columns:
            print(f"Error: Missing required columns - {missing_columns}")
            return False
        
        self.analyze_metadata()
        self.save_report(self.config["data_validation"]["report_save_path"])
        print("Validation completed.")

        return True

# Example Usage
# validator = DataValidator(r"C:\Users\speed\OneDrive\Documents\BITS_Pilani_Mtech\Sem2\DMML\Assignment\Assignment_1\data\intermediate\preproc_data.csv")
# validator.run_validation()
