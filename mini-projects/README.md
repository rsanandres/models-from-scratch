# Mini Projects Collection

A collection of small, experimental projects to explore different technologies, frameworks, and ideas. Each project is self-contained in its own directory and focuses on a specific concept or technology.

## 🚀 Projects

### AI Trend Monitor
A Python-based application that monitors AI trends and implements basic versions of popular AI models. Features include web scraping, trend analysis, and a REST API.

[View Project](ai-trend-monitor/README.md)

### YCombinator Companies Analysis

This project systematically analyzes and documents the top 200 actively hiring YCombinator companies. For each company, it generates either a market research report or a basic AI implementation model, depending on the company's focus.

## 🎯 Purpose

This repository serves as a playground for:
- Learning new technologies
- Experimenting with different frameworks
- Building proof-of-concepts
- Testing ideas quickly
- Having fun with code!

## 📁 Project Structure

Each project follows this general structure:
```
project-name/
├── README.md           # Project-specific documentation
├── requirements.txt    # Python dependencies (if applicable)
├── src/               # Source code
├── tests/             # Test files
└── .gitignore         # Project-specific gitignore
```

## 🛠️ Technologies

Projects in this repository may use various technologies, including but not limited to:
- Python
- JavaScript/TypeScript
- React/Vue/Angular
- Node.js
- Machine Learning frameworks
- Cloud services
- And more!

## 🤝 Contributing

Feel free to:
1. Fork the repository
2. Create a new project directory
3. Add your project with proper documentation
4. Submit a pull request

## 📝 Project Guidelines

When adding a new project:
1. Create a new directory with a descriptive name
2. Include a README.md with:
   - Project description
   - Setup instructions
   - Usage examples
   - Dependencies
3. Add appropriate .gitignore
4. Keep it focused and self-contained

## 📜 License

This repository is open source and available under the MIT License.

## Project Structure

```
yc_companies_analysis/
├── scraper.py              # Main scraper script
├── requirements.txt        # Python dependencies
├── scraper.log            # Log file for scraping operations
└── [Company_Name]/        # Individual company directories
    ├── market_research.md # Market research report
    ├── basic_ai_model.py  # (Optional) Basic AI implementation
    ├── test_ai_model.py   # (Optional) Unit tests
    ├── summary.txt        # Company summary
    └── error_log.txt      # (Optional) Error log
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scraper:
```bash
python scraper.py
```

The script will:
1. Fetch companies from YCombinator's website
2. Create a directory for each company
3. Generate initial market research reports
4. Log any errors encountered

## Features

- Asynchronous web scraping for better performance
- Automatic directory creation and file generation
- Error logging and handling
- Progress tracking with tqdm
- Sanitized file naming

## Output

For each company, the script generates:
- A summary file with basic company information
- A market research report or basic AI model
- Unit tests (if applicable)
- Error logs (if errors occur)

## Contributing

Feel free to submit issues and enhancement requests! 