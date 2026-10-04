"""Download a Google Drive folder into raw_data/<folder name>/, or a single file
(e.g. a Google Sheet, exported as .xlsx) into raw_data/.

Setup: see the steps in the chat / README. Needs credentials.json next to this script.
Usage:
    python download_drive.py <DRIVE_FOLDER_OR_FILE_ID> [local_dir]
    python download_drive.py --verify raw_data/<folder name>   # re-check against manifest
"""
import hashlib
import io
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
HERE = Path(__file__).parent
CREDENTIALS = HERE / "credentials.json"  # downloaded from Google Cloud Console
TOKEN = HERE / "token.json"              # created on first login


def get_service():
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS), SCOPES)
            # WSL: prints a URL to open in your Windows browser
            creds = flow.run_local_server(port=0, open_browser=False)
        TOKEN.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def verify(path, f):
    """Compare the local file's hash with the checksum Drive reports."""
    algo, expected = ("sha256", f.get("sha256Checksum")) if f.get("sha256Checksum") \
        else ("md5", f.get("md5Checksum"))
    if not expected:
        print(f"  no checksum from Drive, can't verify: {path}")
        return True
    return file_hash(path, algo) == expected


def file_hash(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder):
    """{relative path: sha256} for every file under folder, in sorted order."""
    return {p.relative_to(folder).as_posix(): file_hash(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def manifest_path(folder):
    # Kept beside the folder, not inside it, so it doesn't hash itself.
    return folder.with_name(folder.name + ".sha256")


def write_manifest(folder):
    """Write <folder>.sha256 (sha256sum format) and return the hash of the whole folder."""
    hashes = hash_folder(folder)
    text = "".join(f"{h}  {folder.name}/{p}\n" for p, h in hashes.items())
    manifest_path(folder).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_folder(folder):
    """Re-hash folder and compare against its manifest. Returns True if identical."""
    folder = Path(folder).resolve()
    expected = {}
    for line in manifest_path(folder).read_text().splitlines():
        h, p = line.split("  ", 1)
        expected[p.split("/", 1)[1]] = h
    actual = hash_folder(folder)
    changed = [p for p in expected if p in actual and actual[p] != expected[p]]
    missing = [p for p in expected if p not in actual]
    extra = [p for p in actual if p not in expected]
    for label, paths in (("CHANGED", changed), ("MISSING", missing), ("EXTRA", extra)):
        for p in paths:
            print(f"{label}: {p}")
    ok = not (changed or missing or extra)
    print(f"{len(actual)} files checked: " + ("folder matches manifest." if ok else "MISMATCH."))
    return ok


failed = []

FOLDER = "application/vnd.google-apps.folder"
SHEET = "application/vnd.google-apps.spreadsheet"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FIELDS = "id, name, mimeType, md5Checksum, sha256Checksum"


def fetch(request, path):
    with io.FileIO(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def download_file(service, f, path):
    if f["mimeType"] == SHEET:
        # Sheets have no stored file or checksum; export a fresh .xlsx (all tabs) every run.
        path = path.with_name(path.name + ".xlsx")
        print(f"exporting sheet: {path}")
        fetch(service.files().export_media(fileId=f["id"], mimeType=XLSX), path)
    elif f["mimeType"].startswith("application/vnd.google-apps."):
        print(f"skip (Google-native doc): {f['name']}")
    elif path.exists() and verify(path, f):
        print(f"exists, hash OK: {path}")
    else:
        print(f"downloading: {path}")
        fetch(service.files().get_media(fileId=f["id"]), path)
        if verify(path, f):
            print(f"  hash OK")
        else:
            failed.append(path)
            print(f"  HASH MISMATCH: {path}")


def download_folder(service, folder_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields=f"nextPageToken, files({FIELDS})",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        for f in resp["files"]:
            path = dest / f["name"]
            if f["mimeType"] == FOLDER:
                download_folder(service, f["id"], path)
            else:
                download_file(service, f, path)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--verify":
        sys.exit(0 if verify_folder(sys.argv[2]) else 1)
    drive_id = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "raw_data"
    service = get_service()
    f = service.files().get(fileId=drive_id, fields=FIELDS).execute()
    if f["mimeType"] == FOLDER:
        download_folder(service, drive_id, out / f["name"])
    else:
        out.mkdir(parents=True, exist_ok=True)
        download_file(service, f, out / f["name"])
    if failed:
        print(f"\n{len(failed)} file(s) failed hash verification:")
        for p in failed:
            print(f"  {p}")
        sys.exit(1)
    print("\nAll files verified.")
    if f["mimeType"] == FOLDER:
        folder = out / f["name"]
        print(f"Folder hash (sha256): {write_manifest(folder)}")
        print(f"Manifest written to {manifest_path(folder)}")
