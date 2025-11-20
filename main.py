import os
import time
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Business Card OCR → MongoDB",
    page_icon="📇",
    layout="wide",
)

if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
FRONTEND_OPENAI_KEY = os.environ.get("FRONTEND_OPENAI_KEY")  # expected to be set in env (.env) so UI doesn't prompt

st.title("📇 Business Card OCR → MongoDB")
st.write("Upload → Extract OCR (OpenAI required) → Store → Edit → Download")

if not FRONTEND_OPENAI_KEY:
    st.error(
        "FRONTEND_OPENAI_KEY environment variable is not set. "
        "Please set it to a valid OpenAI key (sk-...). The frontend will send this key to the backend for parsing."
    )
    st.stop()

# ----------------------------
# HTTP session & helpers
# ----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(429, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()
DEFAULT_TIMEOUT = 30


def backend_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {FRONTEND_OPENAI_KEY}"} if FRONTEND_OPENAI_KEY else {}


def safe_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BACKEND.rstrip('/')}/{path.lstrip('/')}"
    kwargs.setdefault("timeout", kwargs.pop("timeout", DEFAULT_TIMEOUT))
    try:
        resp = SESSION.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.HTTPError:
        # re-raise so caller can inspect resp if present
        raise
    except Exception as e:
        raise


# ----------------------------
# Utility helpers
# ----------------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


def list_to_csv_str(v):
    if isinstance(v, list):
        return ", ".join([str(x) for x in v])
    return v if v is not None else ""


def csv_str_to_list(s: str):
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _truncate_name(s: str, length: int = 30) -> str:
    if not s:
        return ""
    return s if len(s) <= length else s[: length - 3] + "..."


def _clean_payload_for_backend(payload: dict) -> dict:
    out = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if k in ("phone_numbers", "social_links"):
            if isinstance(v, list):
                out[k] = v
            else:
                out[k] = csv_str_to_list(v)
        else:
            out[k] = v
    return out


# ----------------------------
# Backend actions (thin wrappers)
# ----------------------------
def fetch_all_cards(timeout=20) -> List[Dict[str, Any]]:
    try:
        resp = safe_request("GET", "/all_cards", timeout=timeout, headers=backend_headers())
        data = resp.json()
        return data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        st.error(f"Failed to fetch cards: {e}")
        return []


def patch_card(card_id: str, payload: dict, timeout: int = 30) -> Tuple[bool, str]:
    try:
        resp = safe_request(
            "PATCH",
            f"/update_card/{card_id}",
            json=_clean_payload_for_backend(payload),
            headers=backend_headers(),
            timeout=timeout,
        )
        return True, "Updated" if resp.status_code in (200, 201) else (False, resp.text)
    except requests.HTTPError as he:
        try:
            return False, he.response.json()
        except Exception:
            return False, str(he)
    except Exception as e:
        return False, str(e)


def delete_card(card_id: str, timeout: int = 30) -> Tuple[bool, str]:
    try:
        resp = safe_request("DELETE", f"/delete_card/{card_id}", headers=backend_headers(), timeout=timeout)
        return True, "Deleted" if resp.status_code in (200, 204) else (False, resp.text)
    except requests.HTTPError as he:
        try:
            return False, he.response.json()
        except Exception:
            return False, str(he)
    except Exception as e:
        return False, str(e)


# ----------------------------
# Layout: Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📤 Upload Card", "📁 View All Cards"])

# ----------------------------
# TAB 1 — Upload Card + Manual Form
# ----------------------------
with tab1:
    col_preview, col_upload = st.columns([3, 7])

    with col_upload:
        st.markdown("### Upload card")
        uploaded_file = st.file_uploader(
            "Drag and drop file here\nLimit 200MB • JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file:
            progress = st.progress(10)
            time.sleep(0.05)
            progress.progress(30)
            with st.spinner("Processing image with OCR and uploading..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                try:
                    response = safe_request("POST", "/extract", files=files, headers=backend_headers(), timeout=120)
                except Exception as e:
                    st.error(f"Failed to reach backend: {e}")
                    response = None

                if response is not None:
                    st.write(f"Backend status: {response.status_code}")
                    try:
                        # Sanitize backend's JSON for display (defensive)
                        resp_json = response.json()
                        if isinstance(resp_json, dict) and "data" in resp_json and isinstance(resp_json["data"], dict):
                            inner = dict(resp_json["data"])
                            for remove_key in ("confidence_notes", "extra", "raw_text"):
                                inner.pop(remove_key, None)
                            sanitized = dict(resp_json)
                            sanitized["data"] = inner
                            st.json(sanitized)
                        elif isinstance(resp_json, dict):
                            sanitized = dict(resp_json)
                            for remove_key in ("confidence_notes", "extra", "raw_text"):
                                sanitized.pop(remove_key, None)
                            st.json(sanitized)
                        else:
                            st.text(resp_json)
                    except Exception:
                        st.text(response.text)

                if response and response.status_code in (200, 201):
                    res = response.json()
                    card = res.get("data") if isinstance(res, dict) and "data" in res else res

                    if card:
                        st.success("Extracted — review and save below")
                        card_display = dict(card)
                        # Defensive sanitization - ensure we don't show internal/debug fields
                        for remove_key in ("confidence_notes", "extra", "raw_text"):
                            card_display.pop(remove_key, None)

                        card_display["phone_numbers"] = list_to_csv_str(card_display.get("phone_numbers", []))
                        card_display["social_links"] = list_to_csv_str(card_display.get("social_links", []))

                        df = pd.DataFrame([card_display]).drop(columns=["_id"], errors="ignore")
                        st.dataframe(df, use_container_width=True)

                        if st.button("📥 Save extracted contact to DB"):
                            payload = {
                                "name": card.get("name"),
                                "designation": card.get("designation"),
                                "company": card.get("company"),
                                "phone_numbers": card.get("phone_numbers") or [],
                                "email": card.get("email"),
                                "website": card.get("website"),
                                "address": card.get("address"),
                                "social_links": card.get("social_links") or [],
                                "more_details": card.get("more_details") or "",
                                "additional_notes": card.get("additional_notes") or "",
                            }
                            try:
                                r = safe_request("POST", "/create_card", json=_clean_payload_for_backend(payload), headers=backend_headers(), timeout=30)
                                res2 = r.json()
                                if r.status_code >= 400:
                                    st.error(f"Failed to create card: {res2}")
                                else:
                                    saved = res2.get("data") if isinstance(res2, dict) and "data" in res2 else res2
                                    st.success("Inserted Successfully!")
                                    saved_display = dict(saved)
                                    for remove_key in ("confidence_notes", "extra", "raw_text"):
                                        saved_display.pop(remove_key, None)
                                    saved_display["phone_numbers"] = list_to_csv_str(saved_display.get("phone_numbers", []))
                                    saved_display["social_links"] = list_to_csv_str(saved_display.get("social_links", []))
                                    df2 = pd.DataFrame([saved_display]).drop(columns=["_id"], errors="ignore")
                                    st.dataframe(df2, use_container_width=True)
                                    st.download_button(
                                        "📥 Download as Excel",
                                        to_excel_bytes(df2),
                                        "business_card.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    )
                            except Exception as e:
                                st.error(f"Failed to reach backend: {e}")
                    else:
                        st.warning("Backend returned success but no data payload.")
                else:
                    if response is not None:
                        try:
                            err = response.json()
                        except Exception:
                            err = response.text
                        st.error(f"Upload failed: {err}")
                    else:
                        st.error("Upload failed (no response).")
            progress.progress(100)

    with col_preview:
        st.markdown("### Preview")
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        else:
            st.info("Upload a card to preview here.")

    st.markdown("---")

    with st.expander("📋 Or fill details manually"):
        with st.form("manual_card_form"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Full name")
            designation = c2.text_input("Designation / Title")
            company = c1.text_input("Company")
            phones = c2.text_input("Phone numbers (comma separated)")
            email = c1.text_input("Email")
            website = c2.text_input("Website")
            address = st.text_area("Address")
            social_links = st.text_input("Social links (comma separated)")
            more_details = st.text_area("More details (leave empty to fill later)")
            additional_notes = st.text_area("Notes / extra info")
            submitted = st.form_submit_button("📤 Create Card (manual)")

        if submitted:
            payload = {
                "name": name,
                "designation": designation,
                "company": company,
                "phone_numbers": csv_str_to_list(phones),
                "email": email,
                "website": website,
                "address": address,
                "social_links": csv_str_to_list(social_links),
                "more_details": more_details or "",
                "additional_notes": additional_notes or "",
            }
            with st.spinner("Saving..."):
                try:
                    r = safe_request("POST", "/create_card", json=_clean_payload_for_backend(payload), headers=backend_headers(), timeout=30)
                    res = r.json()
                    if r.status_code >= 400:
                        st.error(f"Failed to create card: {res}")
                    else:
                        created = res.get("data") if isinstance(res, dict) and "data" in res else res
                        if created:
                            st.success("Inserted Successfully!")
                            created_display = dict(created)
                            for remove_key in ("confidence_notes", "extra", "raw_text"):
                                created_display.pop(remove_key, None)
                            created_display["phone_numbers"] = list_to_csv_str(created_display.get("phone_numbers", []))
                            created_display["social_links"] = list_to_csv_str(created_display.get("social_links", []))
                            df = pd.DataFrame([created_display]).drop(columns=["_id"], errors="ignore")
                            st.dataframe(df, use_container_width=True)
                            st.download_button(
                                "📥 Download as Excel",
                                to_excel_bytes(df),
                                "business_card_manual.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                        else:
                            st.warning("Created but no data returned.")
                except Exception as e:
                    st.error(f"Failed to reach backend: {e}")

# ========================================================================
# TAB 2 — View & Edit All Cards
# ========================================================================
with tab2:
    st.markdown("### All business cards")
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.info("Edit any column → press **Save Changes** to apply edits to the backend.")
    with top_col2:
        data = fetch_all_cards()
        if data:
            for d in data:
                # Remove internal/debugging fields before showing or exporting
                d.pop("field_validations", None)
                d.pop("confidence_notes", None)
                d.pop("extra", None)
                d.pop("raw_text", None)
            df_all_for_download = pd.DataFrame(data)
            for col in ["phone_numbers", "social_links"]:
                if col in df_all_for_download.columns:
                    df_all_for_download[col] = df_all_for_download[col].apply(list_to_csv_str)
            st.download_button(
                "📥 Download All as Excel",
                to_excel_bytes(df_all_for_download.drop(columns=["_id"], errors="ignore")),
                "all_business_cards.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.write("")

    with st.spinner("Fetching all business cards..."):
        data = fetch_all_cards()

    if not data:
        st.warning("No cards found.")
    else:
        for d in data:
            d.pop("field_validations", None)
            d.pop("confidence_notes", None)
            d.pop("extra", None)
            d.pop("raw_text", None)

        df_all = pd.DataFrame(data)

        expected_cols = [
            "_id",
            "name",
            "designation",
            "company",
            "phone_numbers",
            "email",
            "website",
            "address",
            "social_links",
            "more_details",
            "additional_notes",
            "created_at",
            "edited_at",
        ]
        for c in expected_cols:
            if c not in df_all.columns:
                df_all[c] = ""

        _ids = df_all["_id"].astype(str).tolist()

        display_df = df_all.copy()
        for col in ["phone_numbers", "social_links"]:
            display_df[col] = display_df[col].apply(list_to_csv_str)

        if "_id" in display_df.columns:
            display_df = display_df.drop(columns=["_id"])

        save_col_left, save_col_mid, save_col_right = st.columns([1, 3, 1])
        with save_col_left:
            save_clicked = st.button("💾 Save Changes")
        with save_col_mid:
            st.write("")
        with save_col_right:
            st.write("")

        try:
            edited = st.experimental_data_editor(display_df, use_container_width=True, num_rows="fixed")
        except Exception:
            edited = st.data_editor(display_df, use_container_width=True, num_rows="fixed")

        if "drawer_open" not in st.session_state:
            st.session_state["drawer_open"] = False
        if "drawer_row" not in st.session_state:
            st.session_state["drawer_row"] = None

        options = []
        for idx, r in df_all.reset_index(drop=True).iterrows():
            display_name = r.get("name") or r.get("company") or r.get("email") or f"Row {idx}"
            options.append(f"{idx} — {display_name}")

        selected = st.selectbox("Select a row to edit", options, index=0, help="Pick a contact to open the edit drawer")

        if st.button("Open selected row in drawer"):
            sel_idx = int(selected.split("—", 1)[0].strip())
            st.session_state["drawer_open"] = True
            st.session_state["drawer_row"] = sel_idx

        if st.session_state.get("drawer_open") and st.session_state.get("drawer_row") is not None:
            sel_idx = st.session_state["drawer_row"]
            if sel_idx < 0 or sel_idx >= len(df_all):
                st.warning("Selected row is no longer available.")
                st.session_state["drawer_open"] = False
                st.session_state["drawer_row"] = None
            else:
                row = df_all.iloc[sel_idx].to_dict()
                id_str = str(row.get("_id"))
                title = f"Edit card — {_truncate_name(row.get('name', ''))}"
                with st.expander(title, expanded=True):
                    c1, c2 = st.columns(2)
                    name_m = c1.text_input("Full name", value=row.get("name", ""), key=f"name-{id_str}")
                    designation_m = c2.text_input("Designation", value=row.get("designation", ""), key=f"desig-{id_str}")
                    company_m = c1.text_input("Company", value=row.get("company", ""), key=f"company-{id_str}")
                    phones_m = c2.text_input(
                        "Phone numbers (comma separated)",
                        value=list_to_csv_str(row.get("phone_numbers", "")),
                        key=f"phones-{id_str}",
                    )
                    email_m = c1.text_input("Email", value=row.get("email", ""), key=f"email-{id_str}")
                    website_m = c2.text_input("Website", value=row.get("website", ""), key=f"website-{id_str}")
                    address_m = st.text_area("Address", value=row.get("address", ""), key=f"address-{id_str}")
                    social_m = st.text_input(
                        "Social links (comma separated)",
                        value=list_to_csv_str(row.get("social_links", "")),
                        key=f"social-{id_str}",
                    )
                    more_m = st.text_area("More details", value=row.get("more_details", ""), key=f"more-{id_str}")
                    notes_m = st.text_area("Notes", value=row.get("additional_notes", ""), key=f"notes-{id_str}")

                    col_ok, col_del, col_close = st.columns([1, 1, 1])
                    with col_ok:
                        if st.button("Save changes", key=f"drawer-save-{id_str}"):

                            def _csv_to_list(s: str):
                                if s is None:
                                    return []
                                return [x.strip() for x in str(s).split(",") if x.strip()]

                            payload = {
                                "name": name_m or None,
                                "designation": designation_m or None,
                                "company": company_m or None,
                                "phone_numbers": _csv_to_list(phones_m),
                                "email": email_m or None,
                                "website": website_m or None,
                                "address": address_m or None,
                                "social_links": _csv_to_list(social_m),
                                "more_details": more_m or None,
                                "additional_notes": notes_m or None,
                            }

                            success, msg = patch_card(id_str, payload)
                            if success:
                                st.success("Updated")
                                st.session_state["drawer_open"] = False
                                st.session_state["drawer_row"] = None
                                st.session_state["refresh_counter"] = st.session_state.get("refresh_counter", 0) + 1
                            else:
                                st.error(f"Failed to update: {msg}")

                    with col_del:
                        if st.button("🗑 Delete card", key=f"drawer-del-{id_str}"):
                            success, msg = delete_card(id_str)
                            if success:
                                st.success("Deleted")
                                st.session_state["drawer_open"] = False
                                st.session_state["drawer_row"] = None
                                st.session_state["refresh_counter"] = st.session_state.get("refresh_counter", 0) + 1
                            else:
                                st.error(f"Failed to delete: {msg}")

                    with col_close:
                        if st.button("Close drawer", key=f"drawer-close-{id_str}"):
                            st.session_state["drawer_open"] = False
                            st.session_state["drawer_row"] = None
                            st.session_state["refresh_counter"] = st.session_state.get("refresh_counter", 0) + 1

        if save_clicked:
            updates = 0
            problems = 0
            for i in range(len(edited)):
                orig = display_df.iloc[i]
                new = edited.iloc[i]

                change_set = {}
                for col in display_df.columns:
                    o = "" if pd.isna(orig[col]) else orig[col]
                    n = "" if pd.isna(new[col]) else new[col]

                    if str(o) != str(n):
                        if col in ["phone_numbers", "social_links"]:
                            items = csv_str_to_list(n)
                            change_set[col] = items
                        else:
                            change_set[col] = n

                if change_set:
                    card_id = _ids[i]
                    success, msg = patch_card(card_id, change_set)
                    if success:
                        updates += 1
                    else:
                        problems += 1
                        st.error(f"Failed to update {card_id}: {msg}")

            if updates > 0:
                st.success(f"✅ Updated {updates} card(s). Refreshing...")
                st.session_state["refresh_counter"] = st.session_state.get("refresh_counter", 0) + 1
            else:
                if problems == 0:
                    st.info("No changes detected.")
                else:
                    st.warning(f"Save completed with {problems} failures.")
