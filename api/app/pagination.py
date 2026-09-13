from dataclasses import dataclass

from fastapi import HTTPException, Query, Request
from sqlalchemy import func, select


@dataclass
class Page:
    number: int
    size: int

    @property
    def offset(self):
        return (self.number - 1) * self.size


def pagination(
    request: Request, page: int = Query(1, ge=1), page_size: int | None = Query(None, ge=1)
):
    settings = request.app.state.settings
    result = Page(page, settings.page_size if page_size is None else page_size)
    if result.size > settings.max_page_size or result.offset > settings.max_offset:
        raise HTTPException(
            422,
            detail={
                "code": "pagination_limit",
                "max_page_size": settings.max_page_size,
                "max_offset": settings.max_offset,
            },
        )
    return result


def envelope(items, total, page):
    return {
        "items": items,
        "total": total,
        "page": page.number,
        "page_size": page.size,
        "pages": (total + page.size - 1) // page.size,
    }


def paginate(db, query, model, page, schema):
    total = db.scalar(select(func.count()).select_from(query.order_by(None).subquery()))
    records = db.scalars(
        query.order_by(model.created_at.desc(), model.id.desc())
        .offset(page.offset)
        .limit(page.size)
    ).all()
    return envelope(
        [schema.model_validate(record).model_dump(mode="json") for record in records], total, page
    )
