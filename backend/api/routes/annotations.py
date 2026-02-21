"""Annotation endpoints: create, list, delete."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db import crud
from schemas.annotations import AnnotationCreate, AnnotationResponse

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


@router.post("", response_model=AnnotationResponse, status_code=201)
def create_annotation(body: AnnotationCreate, db: Session = Depends(get_db)):
    image = crud.get_image(db, body.image_id)
    if not image:
        raise HTTPException(404, f"Image {body.image_id} not found")
    ann = crud.create_annotation(
        db,
        image_id=body.image_id,
        label_class=body.label_class,
        geometry_type=body.geometry_type,
        geometry_json=body.geometry_json,
        notes=body.notes,
    )
    return AnnotationResponse.model_validate(ann)


@router.get("", response_model=list[AnnotationResponse])
def get_annotations(image_id: str, db: Session = Depends(get_db)):
    anns = crud.get_annotations(db, image_id)
    return [AnnotationResponse.model_validate(a) for a in anns]


@router.delete("/{annotation_id}")
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    success = crud.delete_annotation(db, annotation_id)
    if not success:
        raise HTTPException(404, "Annotation not found")
    return {"message": f"Annotation {annotation_id} deleted."}
