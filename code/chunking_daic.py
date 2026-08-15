from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa


INTERRUPT = {373: (395.0, 428.0), 444: (286.0, 387.0), 306: (290.0, 307)}
MISALIGNED = {318: 34.319917, 321: 3.8379167, 341: 6.1892, 362: 16.8582}
EXCLUDED_SESSIONS = {342, 394, 398, 460}
SYNC_LABELS = {"<sync>", "<synch>", "<sync.", "[sync]", "[syncing]"}
EXCEPTION = {
    "<talking to experimenter>",       # 478 — échange avec l'expérimentateur
    "<ellie starts>",                  # 369 — ligne mal-attribuée à Participant, c'est en réalité Ellie
}


def to_ignore(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in SYNC_LABELS:
        return "sync"
    if v in EXCEPTION:
        return "exception"
    if "scrubbed_entry" in v:
        return "scrubbed"
    return None


def load_transcript(path):
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    df = df.rename(columns={"start_time": "start", "stop_time": "stop"})
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["stop"] = pd.to_numeric(df["stop"], errors="coerce")
    df = df.dropna(subset=["start", "stop", "speaker"]).reset_index(drop=True)
    df["speaker"] = df["speaker"].astype(str).str.strip().str.lower()
    return df


def apply_misalignment(df, pid):
    if pid in MISALIGNED:
        offset = MISALIGNED[pid]
        df["start"] += offset
        df["stop"] += offset
    return df


def apply_interrupt(df, pid):
    if pid not in INTERRUPT:
        return df
    inter_start, inter_end = INTERRUPT[pid]
    rows = []
    for _, r in df.iterrows():
        s, e = r["start"], r["stop"]
        if s < inter_start < e:
            rows.append({**r, "stop": inter_start - 0.01})
        elif s < inter_end < e:
            rows.append({**r, "start": inter_end + 0.01})
        elif inter_start <= s and e <= inter_end:
            continue
        else:
            rows.append(r.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def participant_turns(df):
    """Garde uniquement les tours du participant"""
    ellie_idx = df.index[df["speaker"] == "ellie"]
    first_ellie_start = df.loc[ellie_idx[0], "start"] if len(ellie_idx) else 0.0
    df = df[df["start"] >= first_ellie_start]
    return df[df["speaker"] == "participant"].reset_index(drop=True)


def process_session(pid, audio_path, transcript_path, out_dir,
                    save_ignored=False):
    """Nettoie et segmente une session avec un wav par tour de parole participant. Les chunks sont écrits en wav dans out_dir/<pid>/."""
    df = load_transcript(transcript_path)
    df = apply_misalignment(df, pid)
    df = apply_interrupt(df, pid)

    df = df.assign(_ignore_cat=df["value"].apply(to_ignore))
    df_ignored = df[df["_ignore_cat"].notna()].reset_index(drop=True)
    df = df[df["_ignore_cat"].isna()].drop(columns="_ignore_cat").reset_index(drop=True)

    df_keep = participant_turns(df)
    segments = list(zip(df_keep["start"], df_keep["stop"]))

    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if save_ignored:
        for k, r in df_ignored.iterrows():
            cat_dir = Path(out_dir) / "ignored" / r["_ignore_cat"] / str(pid)
            cat_dir.mkdir(parents=True, exist_ok=True)
            i0, i1 = int(r["start"] * sr), int(r["stop"] * sr)
            if i1 > i0:
                sf.write(cat_dir / f"{pid}_{k:04d}.wav", audio[i0:i1], sr)

    pid_dir = Path(out_dir) / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for k, (s, e) in enumerate(segments):
        i0, i1 = int(s * sr), int(e * sr)
        seg = audio[i0:i1]
        if len(seg) == 0:
            continue
        sf.write(pid_dir / f"{pid}_{k:04d}.wav", seg, sr)
        n += 1
    return n


def process_all(input_dir, out_dir, save_ignored=False):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.endswith("_P"))
    for folder in folders:
        pid = int(folder.name.split("_")[0])
        if pid in EXCLUDED_SESSIONS:
            print(f"[skip] {pid} — exclu")
            continue
        audio = folder / f"{pid}_AUDIO.wav"
        transcript = folder / f"{pid}_TRANSCRIPT.csv"
        if not (audio.exists() and transcript.exists()):
            print(f"[skip] {pid} — fichiers manquants")
            continue
        try:
            n = process_session(
                pid, audio, transcript, out_dir,
                save_ignored=save_ignored,
            )
        except Exception as exc:
            print(f"[err ] {pid} — {exc}")
            continue
        print(f"[ok  ] {pid} — {n} chunks")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="dossier contenant les NNN_P/")
    p.add_argument("--out", required=True)
    p.add_argument("--check-ignored", action="store_true",
                   help="sauvegarde aussi les segments ignorés dans out/ignored/<cat>/<pid>/")
    args = p.parse_args()
    process_all(
        input_dir=args.input,
        out_dir=args.out,
        save_ignored=args.check_ignored,
    )
