#!/bin/bash
if [ -f /var/log/audio-env-report.txt ]; then
  cat /var/log/audio-env-report.txt
else
  echo "No setup report found at /var/log/audio-env-report.txt."
fi
