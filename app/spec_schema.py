# spec-writer/app/spec_schema.py
"""
Phase 4: Pydantic models defining the structure of a generated specification.
The LLM is constrained to output data matching these schemas.
"""
from pydantic import BaseModel, Field


class EntityImpact(BaseModel):
    """An entity (class, table, service) affected by the feature."""
    name: str = Field(description="Name of the entity (e.g. TicketService, Ticket table)")
    entity_type: str = Field(description="Type: Class, Interface, Table, Service, Controller, DTO, etc.")
    action: str = Field(description="What happens: CREATE, MODIFY, or REFERENCE")
    details: str = Field(description="Brief description of the change or interaction")


class APIContract(BaseModel):
    """An API endpoint involved in the feature."""
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE")
    path: str = Field(description="Endpoint path, e.g. /api/tickets/{id}")
    description: str = Field(description="What this endpoint does")
    request_body: str = Field(default="", description="Request body shape if applicable")
    response_body: str = Field(default="", description="Response body shape")


class MessageContract(BaseModel):
    """A message/event for async communication (NServiceBus, RabbitMQ, etc.)."""
    name: str = Field(description="Message or event class name")
    direction: str = Field(description="PUBLISH or CONSUME")
    description: str = Field(description="Purpose of this message")
    fields: str = Field(default="", description="Key fields in the message")


class TestCase(BaseModel):
    """A test scenario for the feature."""
    scenario: str = Field(description="What is being tested")
    given: str = Field(description="Preconditions")
    when: str = Field(description="Action performed")
    then: str = Field(description="Expected result")


class Specification(BaseModel):
    """
    Complete specification for a feature, epic, or task.
    Generated from codebase context via RAG + GraphRAG.
    """
    title: str = Field(description="Feature title")
    overview: str = Field(description="High-level description of what this feature does and why")

    entities_affected: list[EntityImpact] = Field(
        default_factory=list,
        description="All classes, tables, services, and interfaces impacted"
    )

    api_contracts: list[APIContract] = Field(
        default_factory=list,
        description="API endpoints to create or modify"
    )

    message_contracts: list[MessageContract] = Field(
        default_factory=list,
        description="Async messages or events involved"
    )

    data_model_changes: str = Field(
        default="",
        description="Database schema changes: new tables, columns, relationships"
    )

    dependencies: str = Field(
        default="",
        description="External services, packages, or systems this feature depends on"
    )

    implementation_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of implementation steps"
    )

    test_cases: list[TestCase] = Field(
        default_factory=list,
        description="Key test scenarios to validate the feature"
    )

    notes: str = Field(
        default="",
        description="Additional considerations, risks, or open questions"
    )