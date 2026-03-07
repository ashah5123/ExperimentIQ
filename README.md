# ExperimentIQ

A statistical experimentation framework for designing, running, and analyzing experiments.

## Project structure

```
ExperimentIQ/
├── api/          # REST API and endpoints
├── core/         # Core experimentation logic and statistics
├── dashboard/    # Dashboard and visualization
├── tests/        # Test suite
├── requirements.txt
├── README.md
└── .env.example
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

3. Copy the example environment file and configure:

   ```bash
   cp .env.example .env
   ```

## Development

Run tests:

```bash
pytest
```

## License

MIT
