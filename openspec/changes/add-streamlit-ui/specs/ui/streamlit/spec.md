## ADDED Requirements

### Requirement: Streamlit Interactive Frontend
The system SHALL provide a Streamlit-based interactive frontend that allows a user to input text and receive a predicted label (spam/ham) and probability from the latest trained model.

#### Scenario: Predict single message
- **WHEN** user pastes or types a message into the UI and clicks predict
- **THEN** the app displays the predicted label and the model's confidence score

#### Scenario: Show explanation
- **WHEN** the model exposes feature importance (e.g., for linear models)
- **THEN** the app SHALL show the top contributing tokens/features and their weights for the prediction

#### Scenario: Health check
- **WHEN** Streamlit app starts
- **THEN** it should be possible to confirm the app is running (a documented local URL and a visible UI)
