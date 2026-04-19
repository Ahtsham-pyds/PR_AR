from collections import defaultdict


def reconcile_claims(claims):
    grouped = defaultdict(list)

    for c in claims:
        key = (c["subject"], c["predicate"], c["object"])
        grouped[key].append(c)

    reconciled = []

    for key, items in grouped.items():
        subject, predicate, obj = key

        # Aggregate confidence
        avg_conf = sum(i["confidence"] for i in items) / len(items)

        reconciled.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": round(avg_conf, 2),
            "sources": list(set(i["source"] for i in items))
        })

    return reconciled