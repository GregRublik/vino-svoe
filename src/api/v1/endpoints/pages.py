# todo эндпоинт выдающий страницу для фотопоиска


from fastapi import APIRouter, Request

from depends import templates


router = APIRouter()

@router.get("/", tags=["pages"])
async def index(
    request: Request,
):
    return templates.TemplateResponse(request, "index.html", {})
