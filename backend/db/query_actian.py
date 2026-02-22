from cortex import CortexClient

def get_slice_id(subject_number: int, slice_index: int, split: str) -> int:
    """Calculate the vector database ID based on subject, slice, and split."""
    base_id = (subject_number - 1) * 135 + (slice_index - 10)
    
    if split.lower() in ("train", "training"):
        return base_id
    elif split.lower() in ("val", "validation"):
        return 49815 + base_id
    else:
        raise ValueError(f"Unknown split: '{split}'. Must be 'training' or 'validation'.")

def query_similar_slices(
    subject_number: int,
    slice_index: int,
    split: str,
    collection_name: str = "breakhis_slices",
    top_n_subjects: int = 10,
    candidates: int = 200,
) -> list[dict]:
    """
    Fetch a slice's vector from the Actian VectorAI collection by ID, search for 
    similar vectors, and return the payload of the best-scoring slice from 
    each of the *top_n_subjects* most similar distinct subjects (excluding the input subject).
    """
    target_id = get_slice_id(subject_number, slice_index, split)

    with CortexClient("100.77.221.91:50051") as client:
        client.open_collection(collection_name)
        
        record = client.get_vector(collection_name=collection_name, id=target_id)
        query_vec = record.vector if hasattr(record, "vector") else record["vector"]
        
        results = client.search(
            collection_name=collection_name,
            query=query_vec,
            top_k=candidates,
            with_payload=True,
        )

    # Pre-populate with the input subject so we skip ALL of their slices
    seen_subjects: set[str] = {str(subject_number)}
    payloads: list[dict] = []

    for r in results:
        payload: dict = r.payload or {}
        
        # Cast to string to safely match the input subject format
        subject_id: str = str(
            payload.get("subject_id")
            or payload.get("patient_id")
            or r.id
        )
        
        # This will now trigger for both the exact input image AND other slices from that subject
        if subject_id in seen_subjects:
            continue
            
        seen_subjects.add(subject_id)
        payloads.append({**payload, "_id": r.id, "_score": r.score})
        
        if len(payloads) >= top_n_subjects:
            break

    return payloads