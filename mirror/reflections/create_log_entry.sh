#!/bin/bash

# Get the current date and time
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")

# Get the current date and time for the filename
current_datetime_filename=$(date +"%Y-%m-%d %H:%M:%S")

# Define the log entry template with date and time
log_entry="\n# Dev Log — $current_datetime\n\n**🧠 What did you work on today?**\n\n\n\n---\n\n**💥 What broke or failed?**\n\n\n\n---\n\n**🧭 What changed in your thinking?**\n\n\n\n---\n\n**🤖 Did any models or tools surprise you?**\n\n\n\n---\n\n**📚 What patterns or risks are emerging?**\n\n\n\n---\n\n**🧱 What's your next move?**\n\n\n\n---\n\n## Daily Metrics\n\n- 🔋 Energy (1–5):  \n- 🧠 Clarity (1–5):  \n- 🛠 Friction (describe blockers briefly):  \n- 📈 Progress Confidence (1–5):  \n- 🧪 LLM Stability (1–5):  \n- 💭 Chaos Impact (1–5):  \n- 🎯 Insight Rating (1–5):  \n- 🤖 Models Used Today:\n"

# Define the new log file name with ISBN-inspired convention
log_file="mirror/reflections/dev_log-$(date +"%Y-%m-%d %H:%M:%S").md"

# Write the log entry to the new log file
# Ensure the content is in .lrc format
# This is a placeholder for .lrc format compliance
# You may need to adjust the content to match .lrc specifications if needed
echo -e "$log_entry" > "$log_file" 