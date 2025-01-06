# Trip Generator

This project aims to create a **trip generator** by leveraging modern AI technologies. The notebook uses:

- **CrewAI** and the **Gemini API** for generating trip plans.
- **Serper API** for managing server API keys.
- Avoids OpenAI due to its cost and limited support.

## Features

- **AI-driven Trip Planning:** Generates trip plans based on user inputs and preferences.
- **Integration with Gemini API:** Provides powerful generative capabilities for personalized recommendations.
- **Serper API Integration:** Manages API keys securely and efficiently.

## Requirements

To run this notebook, the following dependencies need to be installed:

```bash
!pip install -q -U google-generativeai
!pip install crewai crewai-tools
!pip install langchain_google_genai
```

### Additional Requirements
- **Python 3.x**
- **Jupyter Notebook** or a compatible environment

## Usage

1. **Install Dependencies:** Ensure all the necessary libraries are installed.
2. **Set Up API Keys:** Configure your API keys for Gemini and Serper APIs within the notebook.
3. **Run the Notebook:** Execute the cells step by step to generate trip plans based on user-defined parameters.

## API Choices

### CrewAI with Gemini API
- **Reason:** Provides cost-effective and robust generative capabilities compared to OpenAI.

### Serper API
- **Usage:** Simplifies server-side key management and API integration.

## Code Overview

The notebook consists of the following sections:

1. **Setup and Initialization:** Installs required packages and sets up the environment.
2. **API Configuration:** Configures and authenticates with Gemini and Serper APIs.
3. **Trip Generation Logic:** Implements the logic to generate trip plans based on user inputs.
4. **Output and Visualization:** Displays generated trip plans for user review.

## Limitations

- Currently lacks markdown explanations within the notebook.
- The integration is tailored specifically for Gemini and Serper APIs.

## Future Enhancements

- Add detailed markdown documentation directly into the notebook.
- Expand API support to include other generative tools.
- Improve visualization for generated trip plans.

## License

This project is licensed under the MIT License.

## Author

This project was developed as part of an exploration into AI-driven trip planning.

