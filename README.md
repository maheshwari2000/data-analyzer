# LLM Agent for Automated Data Analysis
An AI-powered agent that automates data analysis tasks using Groq, Langchain, and Gradio. This tool takes a CSV dataset as input, performs comprehensive data analysis, and generates a detailed report with insights.

## Features
CSV Data Input: Accepts CSV files via a user-friendly Gradio interface.

Data Analysis Tools:

read_csv_tool: Reads and loads CSV data.

check_data_types_tool: Identifies data types for each column.

check_duplicates_tool: Detects and reports duplicate rows.

clean_data_tool: Cleans and preprocesses data (e.g., handling missing values, formatting).

missing_data_tool: Identifies and summarizes missing data.

stat_data_tool: Generates statistical summaries (e.g., mean, median, standard deviation).

detect_outliers_tool: Detects outliers in the dataset.

correlation_matrix_tool: Computes and visualizes correlation matrices.

Chain-of-Thought Reasoning: Uses Langchain to orchestrate tools and generate detailed analysis reports.

User-Friendly Interface: Built with Gradio for seamless interaction.

## Technologies Used
Python: Primary programming language.

Langchain: For tool orchestration and chain-of-thought reasoning.

Groq: For high-performance LLM inference.

Gradio: For building the web interface.

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Scikit-learn: For outlier detection and statistical analysis.

## How It Works
Input: The user uploads a CSV file via the Gradio interface.

Data Analysis:

The agent reads the CSV file and performs a series of analyses using the integrated tools.

Each tool processes the data and contributes to the final report.

Chain-of-Thought Reasoning: Langchain orchestrates the tools and generates a detailed analysis report.

Output: The user receives a comprehensive report containing:

Data types and summaries.

Duplicate and missing data analysis.

Statistical summaries and outlier detection.

Correlation matrix and insights.

## Example Output
<img width="1067" alt="Screenshot 2025-03-09 at 3 35 16 PM" src="https://github.com/user-attachments/assets/46f2bc06-9984-4d87-b855-5c2929b2b04e" />

Contributions are welcome!
