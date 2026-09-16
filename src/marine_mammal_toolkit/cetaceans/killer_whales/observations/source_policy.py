"""Killer-whale interpretation of provider-specific structured fields."""

from marine_mammal_toolkit.tools.observations.process.adapters import first_present


def twm_evidence(item):
    structured_values = [
        first_present(item.get("pod")),
        first_present(item.get("likelypod")),
        first_present(item.get("pod_tag")),
        first_present(item.get("j_tag")),
        first_present(item.get("k_tag")),
        first_present(item.get("l_tag")),
    ]
    for pod in ("j", "k", "l"):
        flags = (item.get(f"{pod}_pod"), item.get(f"{pod}_likelypod"), item.get(pod))
        if any(
            str(value).strip().lower() in {"1", "1.0", "true", "yes"} for value in flags
        ):
            structured_values.append(f"{pod.upper()} pod")
    return " | ".join(
        dict.fromkeys(
            str(value).strip() for value in structured_values if first_present(value)
        )
    )


TWM_IDENTITY_FIELDS = (
    "sightdate",
    "date",
    "time1",
    "latitude",
    "lat",
    "longitude",
    "lon",
    "pod",
    "likelypod",
    "pod_tag",
    "j_pod",
    "j_likelypod",
    "j",
    "j_tag",
    "k_pod",
    "k_likelypod",
    "k",
    "k_tag",
    "l_pod",
    "l_likelypod",
    "l",
    "l_tag",
    "source",
)
