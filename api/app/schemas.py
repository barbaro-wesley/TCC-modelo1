import re
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Password = Annotated[str, StringConstraints(strip_whitespace=False)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Login(Input):
    email: EmailStr
    password: Password = Field(min_length=1, max_length=128)


class Refresh(Input):
    refresh_token: str = Field(min_length=32, max_length=200)


class NewPassword(Input):
    token: str = Field(min_length=32, max_length=200)
    password: Password = Field(min_length=12, max_length=128)


class UserCreate(Input):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: Password = Field(min_length=12, max_length=128)
    role: Literal["owner", "member", "viewer"] = "member"


class UserUpdate(Input):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    role: Literal["owner", "member", "viewer"] | None = None
    active: bool | None = None


class UserOut(Output):
    id: UUID
    organization_id: UUID | None
    name: str
    email: str
    role: str
    active: bool
    created_at: datetime


class OrganizationCreate(Input):
    name: str = Field(min_length=1, max_length=200)
    cnpj: str
    owner: UserCreate

    @field_validator("cnpj")
    @classmethod
    def validate_cnpj(cls, value):
        value = re.sub(r"[./\-\s]", "", value).upper()
        # Formato numerico e alfanumerico; duas ultimas posicoes sao digitos.
        if not re.fullmatch(r"[A-Z0-9]{12}[0-9]{2}", value) or len(set(value)) == 1:
            raise ValueError("CNPJ invalido")
        base = [ord(c) - 48 for c in value[:12]]
        for weights, digit in (
            ([5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2], value[12]),
            ([6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2], value[13]),
        ):
            remainder = sum(a * b for a, b in zip(base, weights, strict=True)) % 11
            expected = 0 if remainder < 2 else 11 - remainder
            if expected != int(digit):
                raise ValueError("CNPJ invalido")
            base.append(expected)
        return value


class OrganizationUpdate(Input):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    active: bool | None = None


class OrganizationOut(Output):
    id: UUID
    name: str
    cnpj: str
    active: bool
    created_at: datetime


class PlanCreate(Input):
    name: str = Field(min_length=1, max_length=100)
    price_cents: int = Field(ge=0, le=100_000_000)
    monthly_quota: int = Field(ge=0, le=100_000_000)
    rate_per_minute: int = Field(ge=1, le=100_000)
    max_users: int = Field(ge=1, le=10000)
    api_access: bool = False


class PlanOut(Output):
    id: UUID
    name: str
    price_cents: int
    currency: str
    monthly_quota: int
    rate_per_minute: int
    max_users: int
    api_access: bool
    active: bool
    created_at: datetime


class SubscriptionSet(Input):
    plan_id: UUID
    status: Literal["trialing", "active", "past_due", "canceled", "suspended"]
    starts_at: datetime
    ends_at: datetime

    @model_validator(mode="after")
    def dates(self):
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("Datas devem incluir fuso horario")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at deve ser posterior a starts_at")
        return self


class SubscriptionOut(Output):
    id: UUID
    organization_id: UUID
    plan_id: UUID
    status: str
    starts_at: datetime
    ends_at: datetime
    price_cents: int
    monthly_quota: int
    rate_per_minute: int
    max_users: int
    api_access: bool


class KeyCreate(Input):
    label: str = Field(min_length=1, max_length=100)


class KeyOut(Output):
    id: UUID
    label: str
    prefix: str
    active: bool
    created_at: datetime


class UsageOut(Output):
    id: UUID
    user_id: UUID | None
    api_key_id: UUID | None
    request_id: str
    resource: str
    units: int
    period: str
    created_at: datetime


class AuditOut(Output):
    id: UUID
    actor_id: UUID | None
    organization_id: UUID | None
    action: str
    target_id: str
    created_at: datetime


class CostScenario(Input):
    volume_liters: float = Field(gt=0, le=50_000_000, allow_inf_nan=False)
