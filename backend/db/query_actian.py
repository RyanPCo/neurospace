from cortex import CortexClient

def _get_slice_id(subject_number: int, slice_index: int, split: str) -> int:
    """Internal helper: Calculate the vector database ID based on subject, slice, and split."""
    base_id = (subject_number - 1) * 135 + (slice_index - 10)
    
    if split.lower() in ("train", "training"):
        return base_id
    elif split.lower() in ("val", "validation"):
        return 49815 + base_id
    else:
        raise ValueError(f"Unknown split: '{split}'. Must be 'training' or 'validation'.")

def get_similar_subjects_and_slices(
    subject_number: int,
    slice_index: int,
    split: str,
    collection_name: str = "brats_t1_slices",
    top_n_subjects: int = 10,
    candidates: int = 200,
) -> list[dict]:
    """
    Looks up a specific MRI slice in the vector DB and returns the top N most 
    similar slices from *distinct* subjects.
    
    Returns:
        A list of dicts formatted like: [{"subject": 42, "slice": 75}, ...]
    """
    target_id = _get_slice_id(subject_number, slice_index, split)

    # 1. Fetch the query vector and search the database
    with CortexClient("100.77.221.91:50051") as client:
        record = client.get_vector(collection_name=collection_name, id=target_id)
        query_vec = record[0]  # Extracts the vector list from the tuple
        
        results = client.search(
            collection_name=collection_name,
            query=query_vec,
            top_k=candidates,
            with_payload=True,
        )

    # 2. Filter and extract just the subject and slice numbers
    seen_subjects: set[str] = {str(subject_number)}
    final_results: list[dict] = []

    for r in results:
        payload: dict = r.payload or {}
        
        # Safely extract the subject ID
        raw_subject = payload.get("subject_id") or payload.get("patient_id") or r.id
        subject_str = str(raw_subject)
        
        if subject_str in seen_subjects:
            continue
            
        seen_subjects.add(subject_str)
        
        # Safely extract the slice number. 
        # Note: Change "slice_indices" to whatever key you actually used in your payload!
        slice_num = payload.get("slice_indices") or payload.get("slice", -1)
        
        final_results.append({
            "subject": raw_subject,
            "slice": slice_num
        })
        
        if len(final_results) >= top_n_subjects:
            break

    return final_results