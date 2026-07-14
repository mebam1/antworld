#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 GPU_NUMBER [GPU_NUMBER ...]" >&2
  echo "Example: $0 1 2" >&2
}

if (( $# == 0 )); then
  usage
  exit 2
fi

for gpu in "$@"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU number must be a non-negative integer: $gpu" >&2
    usage
    exit 2
  fi
done

for gpu in "$@"; do
  device="/dev/nvidia${gpu}"
  echo "Clearing GPU $gpu ($device)"

  if [[ ! -e "$device" ]]; then
    echo "  Warning: device does not exist; skipping"
    continue
  fi

  declare -A matched_pids=()
  for fd in /proc/[0-9]*/fd/*; do
    link=$(readlink "$fd" 2>/dev/null) || continue
    if [[ "$link" == "$device" ]]; then
      pid=${fd#/proc/}
      pid=${pid%%/fd/*}
      matched_pids["$pid"]=1
    fi
  done

  if (( ${#matched_pids[@]} == 0 )); then
    echo "  No process found"
    continue
  fi

  for pid in "${!matched_pids[@]}"; do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs) || continue
    if [[ "$owner" == "$USER" ]]; then
      process_name=$(ps -o comm= -p "$pid" 2>/dev/null | xargs)
      echo "  Killing PID $pid (${process_name:-unknown})"
      kill -9 "$pid"
    else
      echo "  Skipping PID $pid owned by ${owner:-unknown}"
    fi
  done

  unset matched_pids
done
