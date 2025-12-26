from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ProjectEntity(BaseModel):
    """Проект с компонентами и статусом."""

    name: str = Field(description="Название проекта")
    status: str = Field(
        description="Статус: Concept, Development, Testing, Production, Archived",
        default="Development",
    )
    components: List[str] = Field(
        description="Список компонентов проекта",
        default_factory=list,
    )
    owner: str = Field(
        description="Владелец/lead проекта",
        default="Unknown",
    )
    priority: int = Field(
        description="Приоритет: 1-Critical, 2-High, 3-Medium, 4-Low",
        default=3,
    )


class TechnicalConceptEntity(BaseModel):
    """Техническая концепция или архитектурный паттерн."""

    name: str = Field(description="Название концепции (Fractal, Graph, Memory, etc)")
    description: str = Field(description="Краткое описание концепции")
    abstraction_level: int = Field(
        description="Уровень: 1-Basic, 2-Intermediate, 3-Advanced, 4-Research",
        default=2,
    )
    related_concepts: List[str] = Field(
        description="Связанные концепции",
        default_factory=list,
    )
    implementation_status: str = Field(
        description="Статус реализации: Theoretical, Prototype, Production-Ready",
        default="Theoretical",
    )


class DecisionEntity(BaseModel):
    """Решение, которое может быть переоценено."""

    decision_text: str = Field(description="Формулировка решения")
    decision_date: datetime = Field(description="Когда было принято решение")
    decision_maker: str = Field("Кто принял решение")
    rationale: str = Field("Причины, по которым было принято решение")
    status: str = Field(
        description="Статус: Active, Superseded, Rejected, Pending-Review",
        default="Active",
    )
    dependencies: List[str] = Field(
        description="На что влияет это решение",
        default_factory=list,
    )


class TeamEntity(BaseModel):
    """Команда или группа людей."""

    team_name: str = Field(description="Название команды")
    members: List[str] = Field(description="Члены команды")
    focus: str = Field(description="На чём фокусируется команда")
    communication_tool: Optional[str] = Field(
        description="Инструмент общения (Telegram, Slack, Discord)",
        default=None,
    )


class L3Summary(BaseModel):
    """Высокоуровневая абстракция или резюме группы эпизодов памяти."""
    summary_text: str = Field(description="Резюме или абстрактный вывод")
    consolidated_from: List[str] = Field(
        description="UUID эпизодов, на основе которых сделана абстракция",
        default_factory=list,
    )
    generated_at: datetime = Field(
        description="Когда была сгенерирована эта абстракция",
        default_factory=lambda: datetime.now(timezone.utc),
    )
    group_id: str = Field(description="Группа памяти (e.g., personal, project, knowledge)")