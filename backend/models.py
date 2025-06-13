from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from datetime import datetime

class TankCondition(BaseModel):
    tank_id: str
    product: str
    api: float
    ullage_ft: float
    ullage_in: float
    temperature_f: float
    water_bbls: float = 0.0
    gross_bbls: float
    net_bbls: float | None = Field(default=None)
    metric_tons: float | None = Field(default=None)

    model_config = ConfigDict(extra='forbid')

class ProductTotals(BaseModel):
    gross_bbls: float
    net_bbls: float | None = Field(default=None)
    metric_tons: float | None = Field(default=None)

    model_config = ConfigDict(extra='forbid')

class Timestamps(BaseModel):
    arrival: Optional[datetime] = None
    all_fast: Optional[datetime] = None
    boom_on: Optional[datetime] = None
    hose_on: Optional[datetime] = None
    comm_ld: Optional[datetime] = None
    comp_ld: Optional[datetime] = None
    hose_off: Optional[datetime] = None
    boom_off: Optional[datetime] = None
    depart: Optional[datetime] = None

    model_config = ConfigDict(extra='forbid')

class Drafts(BaseModel):
    fwd_port: float
    fwd_stbd: float
    aft_port: float
    aft_stbd: float

    model_config = ConfigDict(extra='forbid')

class BargeInfo(BaseModel):
    name: str
    voyage_number: Optional[str] = None
    otb_job_number: Optional[str] = None

    model_config = ConfigDict(extra='forbid')

class PortInfo(BaseModel):
    vessel_name: str
    port_city: Optional[str] = None

    model_config = ConfigDict(extra='forbid')

class ArrivalDeparture(BaseModel):
    water_specific_gravity: Optional[float] = None
    drafts_ft: Optional[Drafts] = None
    timestamps: Optional[Timestamps] = None
    tanks: List[TankCondition]
    summary_by_product: Optional[Dict[str, ProductTotals]] = None

    model_config = ConfigDict(extra='forbid')

class FieldDocument(BaseModel):
    barge: BargeInfo
    port: PortInfo
    arrival: ArrivalDeparture
    departure: ArrivalDeparture
    products_loaded_discharged: Optional[Dict[str, ProductTotals]] = None

    model_config = ConfigDict(extra='forbid')
