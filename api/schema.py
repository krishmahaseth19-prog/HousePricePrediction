from pydantic import BaseModel


class HouseData(BaseModel):

    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    city: str
    statezip: str
    country: str
    year: int
    month: int