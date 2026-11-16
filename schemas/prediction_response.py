from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    predicted_price: float = Field(
        ...,
        description="Predicted market price of the car in Pakistani Rupees",
        examples=[6579612.5]
    )

    model_version: str = Field(
        ...,
        description="Version of the machine learning model",
        examples=["1.0.0"]
    )

    currency: str = Field(
        ...,
        description="Currency of the predicted price",
        examples=["PKR"]
    )

    message: str = Field(
        ...,
        description="Prediction status message",
        examples=["Car price predicted successfully"]
    )