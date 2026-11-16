from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated


class UserInput(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    nam: Annotated[
        str,
        Field(
            ...,
            description="Name/model of the car",
            examples=["Toyota Corolla Altis"]
        )
    ]

    Year: Annotated[
        int,
        Field(
            ...,
            description="Manufacturing/model year of the car",
            examples=[2014]
        )
    ]

    Fuel: Annotated[
        str,
        Field(
            ...,
            description="Fuel type of the car",
            examples=["Petrol"]
        )
    ]

    Transmission: Annotated[
        str,
        Field(
            ...,
            description="Transmission type",
            examples=["Automatic"]
        )
    ]

    Province: Annotated[
        str,
        Field(
            ...,
            description="Province where the car is located",
            examples=["Punjab"]
        )
    ]

    Color: Annotated[
        str,
        Field(
            ...,
            description="Exterior color of the car",
            examples=["White"]
        )
    ]

    Assembly: Annotated[
        str,
        Field(
            ...,
            description="Assembly type of the car",
            examples=["Imported"]
        )
    ]

    Body_Type: Annotated[
        str,
        Field(
            ...,
            alias="Body Type",
            description="Body type of the car",
            examples=["Sedan"]
        )
    ]

    Features: Annotated[
        list[str],
        Field(
            ...,
            description="Features available in the car",
            examples=[
                [
                    "Air Conditioning",
                    "Power Steering",
                    "Power Windows"
                ]
            ]
        )
    ]

    Millage_km: Annotated[
        float,
        Field(
            ...,
            description="Mileage of the car in kilometers",
            examples=[43000.0]
        )
    ]

    Engine_Value: Annotated[
        float,
        Field(
            ...,
            description="Engine capacity/value",
            examples=[1360.0]
        )
    ]

    Engine_Unit: Annotated[
        str,
        Field(
            ...,
            description="Unit of engine capacity",
            examples=["cc"]
        )
    ]


