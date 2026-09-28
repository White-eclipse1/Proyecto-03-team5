import {
  type Annotation,
  annotationSchema,
  annotationsResponseSchema,
  type CreateAnnotationBody,
  createAnnotationBodySchema,
  type PatchAnnotationBody,
  patchAnnotationBodySchema,
} from "../../types/schemas";
import { apiRequest, apiRequestVoid, jsonBody } from "./client";

export function getAnnotations(imageId: number): Promise<Annotation[]> {
  return apiRequest(`/images/${imageId}/annotations`, annotationsResponseSchema);
}

export function createAnnotation(imageId: number, body: CreateAnnotationBody): Promise<Annotation> {
  const validBody = createAnnotationBodySchema.parse(body);
  return apiRequest(`/images/${imageId}/annotations`, annotationSchema, {
    method: "POST",
    ...jsonBody(validBody),
  });
}

export function updateAnnotation(
  annotationId: number,
  body: PatchAnnotationBody
): Promise<Annotation> {
  const validBody = patchAnnotationBodySchema.parse(body);
  return apiRequest(`/annotations/${annotationId}`, annotationSchema, {
    method: "PATCH",
    ...jsonBody(validBody),
  });
}

export function deleteAnnotation(annotationId: number): Promise<void> {
  return apiRequestVoid(`/annotations/${annotationId}`, { method: "DELETE" });
}
