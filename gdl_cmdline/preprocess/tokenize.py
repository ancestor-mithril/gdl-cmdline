import base64
import ctypes
import gzip
import os
import warnings
import zlib

import regex as re

IPV4SEG = r"(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
IPV4ADDR = r"(?:(?:" + IPV4SEG + r"\.){3,3}" + IPV4SEG + r")"

IPV6SEG = r"(?:(?:[0-9a-fA-F]){1,4})"
IPV6GROUPS = (
    r"(?:" + IPV6SEG + r":){7,7}" + IPV6SEG,
    r"(?:" + IPV6SEG + r":){1,7}:",
    r"(?:" + IPV6SEG + r":){1,6}:" + IPV6SEG,
    r"(?:" + IPV6SEG + r":){1,5}(?::" + IPV6SEG + r"){1,2}",
    r"(?:" + IPV6SEG + r":){1,4}(?::" + IPV6SEG + r"){1,3}",
    r"(?:" + IPV6SEG + r":){1,3}(?::" + IPV6SEG + r"){1,4}",
    r"(?:" + IPV6SEG + r":){1,2}(?::" + IPV6SEG + r"){1,5}",
    IPV6SEG + r":(?:(?::" + IPV6SEG + r"){1,6})",
    r":(?:(?::" + IPV6SEG + r"){1,7}|:)",
    r"fe80:(?::" + IPV6SEG + r"){0,4}%[0-9a-zA-Z]{1,}",
    r"::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]" + IPV4ADDR,
    r"(?:" + IPV6SEG + r":){1,4}:[^\s:]" + IPV4ADDR,
)
IPV6ADDR = "|".join(["(?:{})".format(g) for g in IPV6GROUPS[::-1]])

# IPv4 Internal: 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x
IPV4_INTERNAL_PREFIX = (
    r"(?:127\.|10\.|172\.(?:1[6-9]|2[0-9]|3[0-1])\.|192\.168\.|169\.254\.)"
)

# IPv6 Internal: fe80: (Link Local), fc00/fd00 (Unique Local), ::1 (Loopback)
# Note for ::1: We ensure it is NOT followed by a hex char (to avoid matching ::1234)
IPV6_INTERNAL_PREFIX = r"(?:fe80:|f[cd][0-9a-fA-F]{2}:|::1(?![0-9a-fA-F]))"


# Logic: Lookahead asserts it starts with Internal Prefix, THEN match the IP
ipv4_internal_pattern = re.compile(
    r'(?i)["\']?(?<![a-zA-Z0-9])(?:localhost|(?='
    + IPV4_INTERNAL_PREFIX
    + r")"
    + IPV4ADDR
    + r')\b["\']?'
)

# --- EXTERNAL IPv4 ---
# Logic: Negative Lookahead asserts it does NOT start with Internal Prefix, THEN match the IP
ipv4_external_pattern = re.compile(
    r'(?i)["\']?(?<![a-zA-Z0-9])(?!'
    + IPV4_INTERNAL_PREFIX
    + r")"
    + IPV4ADDR
    + r'\b["\']?'
)

# --- INTERNAL IPv6 ---
ipv6_internal_pattern = re.compile(
    r'(?i)["\']?(?<![a-zA-Z0-9])(?='
    + IPV6_INTERNAL_PREFIX
    + r")(?:"
    + IPV6ADDR
    + r')(?![a-zA-Z0-9])["\']?'
)

# --- EXTERNAL IPv6 ---
ipv6_external_pattern = re.compile(
    r'(?i)["\']?(?<![a-zA-Z0-9])(?!'
    + IPV6_INTERNAL_PREFIX
    + r")(?:"
    + IPV6ADDR
    + r')(?![a-zA-Z0-9])["\']?'
)

ssh_pattern = re.compile(r"\.ssh([\\/]+)[^\\/\s]+")


# DOS 8.3 short name mappings for known Windows system directories
# These are expanded FIRST before other normalizations
# PROGRA~1/~2 = Program Files or Program Files (x86)
# PROGRA~3 = ProgramData
# WINDOW~1 = Windows
DOS_SHORTNAME_MAP = {
    r"\bC:[/\\]+PROGRA~[12][/\\]+": r"C:\\Program Files\\",
    r"\bC:[/\\]+PROGRA~3[/\\]+": r"C:\\ProgramData\\",
    r"\bC:[/\\]+WINDOW~1[/\\]+": r"C:\\Windows\\",
}

# Generic DOS 8.3 short name pattern - collapses NAME~1, NAME~2, etc. to just NAME
# This is applied AFTER specific expansions to clean up remaining short names
dos_shortname_generic = re.compile(r"([A-Za-z0-9_]+)~+\d+", re.IGNORECASE)


def make_windows_path_pattern(base_path, skip_one_dir=False, optional_suffix=""):
    r"""
    Create a regex pattern for matching Windows C: drive paths.
    Handles both Windows style (C:\) and Unix style (/c/) paths.

    Args:
        base_path: Base directory name (e.g., "Program Files", "ProgramData", "Windows", "Users")
        skip_one_dir: If True, skips one directory level (e.g., for Users\username\...)
        optional_suffix: Optional suffix to add after base_path (e.g., r'(?:\s\(x86\))?' for Program Files)

    Returns:
        Compiled regex pattern that:
        - Matches C:\ or C:/ or /c/ with the base path
        - Captures everything after base path (and optional skipped dir)
        - Allows spaces in paths
        - Stops at quotes, pipes, redirections, or command flags
        - Stops at flag-like patterns (e.g., " /g" or " -h") unless followed by extension/path
    """
    # Build the pattern parts
    skip_segment = r"[^/\\]+(?:[/\\]+|$)" if skip_one_dir else ""

    # Flag-like pattern: space + flag prefix + word, NOT followed by extension or path separator
    # This detects flags like " /g", " -h", " -help", " --verbose" which are NOT paths
    # Pattern: whitespace + (/ or - or --) + letter + optional more flag chars
    # NOT followed by . (extension) or / or \ (more path)
    flag_lookahead = r"\s(?:/|--?)[a-zA-Z][-a-zA-Z0-9]*(?![./\\])"

    # Four branches: immediately after quote, inside quotes (not immediately), or unquoted
    pattern = (
        r"(?:"
        # Branch 1: Double-quoted path (immediately preceded by ") - allow spaces inside
        + r'(?<=")(?:C:|/c)[/\\]+'  # Preceded by ", then C:\ or /c/
        + re.escape(base_path)
        + optional_suffix
        + r"[/\\]+"
        + skip_segment
        + r'([^"<>|&:]?)'  # Capture: allow spaces, stop at special chars
        + r'(?="|&|'
        + flag_lookahead
        + r"|$)"  # Lookahead: end at " OR flag-like pattern
        + r"|"
        # Branch 2: Single-quoted path (immediately preceded by ') - allow spaces inside
        + r"(?<=')(?:C:|/c)[/\\]+"  # Preceded by ', then C:\ or /c/
        + re.escape(base_path)
        + optional_suffix
        + r"[/\\]+"
        + skip_segment
        + r"([^'<>|&:]*?)"  # Capture: allow spaces, stop at special chars
        + r"(?='|&|"
        + flag_lookahead
        + r"|$)"  # Lookahead: end at ' OR flag-like pattern
        + r"|"
        # Branch 3: Path inside double-quotes (not immediately after ") - allow spaces
        # Handles: "cmd /c C:\Program Files\..." where path is after other content
        + r"(?<= )(?:C:|/c)[/\\]+"  # Preceded by space
        + re.escape(base_path)
        + optional_suffix
        + r"[/\\]+"
        + skip_segment
        + r'([^"<>|&:]*?)'  # Capture: allow spaces
        + r'(?="|&|'
        + flag_lookahead
        + r"|$)"  # Lookahead: end at " OR flag-like pattern
        + r"|"
        # Branch 4: Unquoted path - stop at whitespace
        + r"(?<![\"'])(?:C:|/c)[/\\]+"  # NOT preceded by quote, then C:\ or /c/
        + re.escape(base_path)
        + optional_suffix
        + r"[/\\]+"
        + skip_segment
        + r'([^"<>|&:\s]*)'  # Capture: no spaces allowed
        + r"(?=\s|[\"']|$)"  # Lookahead: stop at whitespace, quote, or end
        + r")"
    )

    return re.compile(pattern, re.IGNORECASE)


# Compiled patterns for C: drive special directories
# Note: These patterns allow spaces in paths and stop at appropriate delimiters
program_files = make_windows_path_pattern(
    "Program Files", optional_suffix=r"(?:\s\(x86\))?"
)
program_data = make_windows_path_pattern("ProgramData")
users_path = make_windows_path_pattern(
    "Users", skip_one_dir=True
)  # Skip username directory
windows_path = make_windows_path_pattern("Windows")

# Generic C: drive pattern for any other directories (applied LAST after specific patterns)
# Matches: C:\OEM\..., C:\managecore\..., /c/tools\..., C:\any_other_dir\...
# Excludes: Program Files, ProgramData, Users, Windows, Temp (handled by specific patterns above)
# Format: C:\FirstDir\path\to\file → <C>\FirstDir\<path>\file
#         /c/FirstDir/path/to/file → <C>\FirstDir\<path>\file
# Two branches: quoted (allows spaces) and unquoted (no spaces)
c_drive_generic = re.compile(
    r"(?:"
    # Branch 1: Double-quoted path (preceded by ") - allow spaces inside, stop at closing quote
    + r'(?<=")(?:C:|/c|\\mnt\\c)[/\\]+'  # Preceded by ", then C:\ or /c/
    + r"(?!"  # Negative lookahead: exclude these:
    r"(?:Program Files|ProgramData|Users|Windows|Temp)\b"
    r")"
    + r'([^"<>|&]+?)'  # Capture group 1: path with spaces
    + r'(?=")'  # Stop at closing double quote
    + r"|"
    # Branch 2: Single-quoted path (preceded by ') - allow spaces inside, stop at closing quote
    + r"(?<=')(?:C:|/c|\\mnt\\c)[/\\]+"  # Preceded by ', then C:\ or /c/
    + r"(?!"  # Negative lookahead: exclude these:
    r"(?:Program Files|ProgramData|Users|Windows|Temp)\b"
    r")"
    + r"([^'<>|&]+?)"  # Capture group 2: path with spaces
    + r"(?=')"  # Stop at closing single quote
    + r"|"
    # Branch 3: Unquoted path - no spaces allowed
    + r"(?<![\"'])(?:C:|/c|\\mnt\\c)[/\\]+"  # NOT preceded by quote, then C:\ or /c/
    + r"(?!"  # Negative lookahead: exclude these:
    r"(?:Program Files|ProgramData|Users|Windows|Temp)\b"
    r")"
    + r'([^\s"\'<>|&]+)'  # Capture group 3: path without spaces
    + r"(?=\s|[\"']|$)"  # Stop at whitespace, quote, or end
    + r")",
    re.IGNORECASE,
)

# Environment variable paths: %VAR%\path\to\file → %VAR%\<path>\file
# Matches: %SystemRoot%, %USERPROFILE%, %TEMP%, etc.
env_var_path = re.compile(
    r"(%[A-Za-z_][A-Za-z0-9_]*%)"  # Capture group 1: %VARNAME%
    r"[/\\]+"  # Path separator
    r'([^\s"<>|&\)]+)',  # Capture group 2: rest of path (stop at special chars, including ))
    re.IGNORECASE,
)

# UNC paths with IP addresses: \\<ipv4>\share\path\file → \\<ipv4>\<path>\file
# Applied AFTER IP addresses are replaced with <ipv4>/<ipv6> tags
# Two branches: quoted (allows spaces) and unquoted (no spaces)
unc_ip_path = re.compile(
    r"(?:"
    # Branch 1: Quoted UNC path - allow spaces
    + r'(?<=")\\\\<int_ip>'  # Preceded by ", then \\<int_ip> or \\<ext_ip>
    + r"[/\\]+"  # Path separator after IP
    + r'([^"<>|&]+?)'  # Capture group 1: path with spaces
    + r'(?=")'  # Stop at closing quote
    + r"|"
    # Branch 2: Unquoted UNC path - no spaces
    + r'(?<!["])\\\\<int_ip>'  # NOT preceded by ", then \\<ipv4> or \\<ipv6>
    + r"[/\\]+"  # Path separator after IP
    + r'([^\s"<>|&]+)'  # Capture group 2: path without spaces
    + r"|"
    # Branch 3: Quoted UNC path - allow spaces
    + r'(?<=")\\\\<ext_ip>'  # Preceded by ", then \\<int_ip> or \\<ext_ip>
    + r"[/\\]+"  # Path separator after IP
    + r'([^"<>|&]+?)'  # Capture group 1: path with spaces
    + r'(?=")'  # Stop at closing quote
    + r"|"
    # Branch 4: Unquoted UNC path - no spaces
    + r'(?<!["])\\\\<ext_ip>'  # NOT preceded by ", then \\<ipv4> or \\<ipv6>
    + r"[/\\]+"  # Path separator after IP
    + r'([^\s"<>|&]+)'  # Capture group 2: path without spaces
    + r")",
    re.IGNORECASE,
)

# UNC paths with hostnames: \\hostname\share\path\file → \\<host>\<path>\file
# Matches hostnames like: server01, infra-ab-dc01.psi.de, etc.
unc_host_path = re.compile(
    r"(?:"
    # Branch 1: Quoted UNC path - allow spaces
    + r'(?<=")(?:\\\\\\\\|\\\\)([a-zA-Z0-9][-a-zA-Z0-9.]*)'  # Preceded by ", then \\hostname
    + r"[/\\]+"  # Path separator after hostname
    + r'([^"<>|&]+?)'  # Capture group 2: path with spaces
    + r'(?=")'  # Stop at closing quote
    + r"|"
    # Branch 2: Unquoted UNC path - no spaces
    + r'(?<![:"a-z0-9\\])(?:\\\\\\\\|\\\\)([a-zA-Z0-9][-a-zA-Z0-9.]*)'  # NOT preceded by " or :, then \\hostname
    + r"[/\\]+"  # Path separator after hostname
    + r'([^\s"<>|&]+)'  # Capture group 4: path without spaces
    + r")",
    re.IGNORECASE,
)

# Matches both normalized paths (<path>\.vcxproj) and standalone filenames (MyProject.vcxproj)
vcxproj_files = re.compile(r"(?:<path>\\)?\.vcxproj\b|\b[^\s\\\/]+\.vcxproj\b")
vs_version = re.compile(r"VisualStudioVersion=\d+\.\d+", re.IGNORECASE)
vs_configuration = re.compile(r"Configuration=\d+", re.IGNORECASE)
vs_platform = re.compile(r"Platform=\d+", re.IGNORECASE)

# MSVC linker/lib patterns
# /out:path\file.lib or /out:path\file.exe → /out:<path>\.lib or /out:<path>\.exe
msvc_out_pattern = re.compile(r'/out:([^\s"]+)', re.IGNORECASE)
# Object files: path\file.obj → <path>\.obj (relative paths ending in .obj)
msvc_obj_pattern = re.compile(r'(?<=\s)([^\s"<>]+\.obj)\b', re.IGNORECASE)

# Unix/Linux standard directories
# Matches: /tmp, /etc, /usr, /var, /opt, /home
# Format: /base/first_dir/.../file → /base/first_dir/<path>/file
# For /tmp: simplified to <tmp>.ext
# Handles both clean paths and escaped quotes: \""path\""
tmp_files_linux = re.compile(r'(^|\s|[\\"]*")(/tmp/\S+)\.(\w+)(?=[\\"]*"|\s|$)')
tmp_files_linux_no_ext = re.compile(r'(^|\s|[\\"]*")(/tmp/\S+)(?=[\\"]*"|\s|$)')

# Standard Unix directories (etc, usr, var, opt, home, bin, sbin, lib)
# Format: /base/first/.../file → <path>\file
# Handles both clean paths and paths with escaped quotes: \""path\""
# Negative lookbehind prevents matching URLs or relative paths
unix_standard_dirs = re.compile(
    r'(?:[\\"]*")?'  # Optional: escaped quotes at start (\""
    r"(?<![a-z0-9./])"  # Not preceded by alphanumeric, dot, or slash
    r"/(etc|usr|var|opt|home|bin|sbin|lib)"  # Capture group 1: Base directory
    r'(/[^\s"\'\\:;|&]+)?'  # Capture group 2: Everything after base (optional)
    r'(?:[\\"]*")?'  # Optional: escaped quotes at end \""
    r'(?=\s|"|\'|$|[;|&])',  # Stop at whitespace, quote, or shell special chars
    re.IGNORECASE,
)

# Windows temp directories - split into separate patterns for speed
# Each pattern is simple and non-backtracking
# Format: C:\Temp\... or /c/Temp/... → <tmp>.ext or <tmp>

# C:\Temp\file.txt or /c/Temp/file.txt → <tmp>.txt
c_temp_ext = re.compile(
    r'(?:C:|/c)[/\\]+Temp[/\\]+[^"<>|&:?*]*?\.(\w+)(?="|$|\s)', re.IGNORECASE
)
c_temp_no_ext = re.compile(
    r'(?:C:|/c)[/\\]+Temp[/\\]+[^"<>|&.:?*]+(?="|$|\s)', re.IGNORECASE
)

# C:\Windows\Temp\file.txt or /c/Windows/Temp/file.txt → <tmp>.txt
windows_temp_ext = re.compile(
    r'(?:C:|/c)[/\\]+Windows[/\\]+Temp[/\\]+[^"<>|&:?*]*?\.(\w+)(?="|$|\s)', re.IGNORECASE
)
windows_temp_no_ext = re.compile(
    r'(?:C:|/c)[/\\]+Windows[/\\]+Temp[/\\]+[^"<>|&.:?*]+(?="|$|\s)', re.IGNORECASE
)

# C:\Users\username\AppData\Local\Temp\file.txt or /c/Users/... → <tmp>.txt
users_temp_ext = re.compile(
    r'(?:C:|/c)[/\\]+Users[/\\]+[^/\\]+[/\\]+AppData[/\\]+Local[/\\]+Temp[/\\]+[^"<>|&]*?\.(\w+)(?="|$|\s)',
    re.IGNORECASE,
)
users_temp_no_ext = re.compile(
    r'(?:C:|/c)[/\\]+Users[/\\]+[^/\\]+[/\\]+AppData[/\\]+Local[/\\]+Temp[/\\]+[^"<>|&.]+(?="|$|\s)',
    re.IGNORECASE,
)

# Network address patterns (with word boundaries for command context)
# Includes localhost as an IPv4 equivalent, with optional surrounding quotes

# MAC address: supports both colon (aa:bb:cc:dd:ee:ff) and dash (aa-bb-cc-dd-ee-ff) separators
# With optional surrounding quotes
mac_address_pattern = re.compile(
    r'["\']?\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b["\']?'
)

# IP:port patterns - match after IP substitution to catch <ipv4>:1234 or 1234:<ipv4>:5678
ip_port_pattern = re.compile(r"<(int|ext)_ip>:(\d+)")  # <ipv4>:1234
url_port_pattern = re.compile(r"<url>:(\d+)")  # <url>:1234
ip_port_pattern2 = re.compile(r"(\d+):<(int|ext)_ip>:(\d+)")  # 1234:<ipv4>:5678
ip_comma_port_pattern = re.compile(
    r"<(int|ext)_ip>,(\d+)"
)  # <ipv4>,50288 (tcp:ip,port format)
# tcp:hostname,port pattern: tcp:USAS2-IMCHMIP,1433 → tcp:<host>,<port>
tcp_host_port_pattern = re.compile(
    r"(tcp:)([A-Za-z][A-Za-z0-9_-]*),(\d+)", re.IGNORECASE
)

# Unix-like paths on Windows (Cygwin, Git Bash, WSL, etc.)
# Matches: /cygdrive/d/path/to/file, /d/path/file.txt, /c/bin, /mnt/c/path
# Stops at: spaces, quotes, backslash, colon, pipe, ampersand, semicolon, etc.
# Captures the final filename/directory
# Uses negative lookbehind to avoid matching URLs or relative paths
# Requires at least ONE slash after drive letter to avoid matching command flags like /d, /c
unix_path_on_windows = re.compile(
    r'(?<![a-zA-Z0-9.])(?:/cygdrive/[a-z](?:/[^\s"\'\\:;|&/]+)+|/mnt/[a-z](?:/[^\s"\'\\:;|&/]+)+|/[a-z](?:/[^\s"\'\\:;|&/]+){2,})/([^\s"\'\\:;|&/]+)',
    re.IGNORECASE,
)

# Windows paths on non-C drives (D:, E:, F:, etc.)
# Matches: D:\path\to\file.txt, E:/folder/file.exe, /d/path/file, /e/folder/file (mixed slashes)
# Captures everything after the drive letter for hierarchical normalization
# Format: D:\FirstDir\...\file → <path>\FirstDir\<path>\.ext
#         /d/FirstDir/.../file → <path>\FirstDir\<path>\.ext
# Lookahead stops at: quote, end, flags (-/), shell operators (&&, ||, |, ;), UNC paths (\\)
windows_other_drives = re.compile(
    r'(?:\b[A-BD-Z]:|/[a-bd-z])[/\\]+([^"<>|&:?*]+?)(?="|$|\s[-/]|\s/|\s\w+\s[-/]|\s*&&|\s*\|\||\s*\||\s*;|\s\\\\)',
    re.IGNORECASE,
)

flag_cluster_pattern = re.compile(r"(?:(?<=\s)|(?<=^))(?:/[a-zA-Z]){2,}\b")

has_timestamp_pattern = re.compile(r"\b([0-9]{10}|[0-9]{13})\b")

# Windows SID pattern
# Must be matched BEFORE timestamp to avoid partial matches
# Negative lookbehind ensures S is not preceded by a letter (e.g., "AS-1-5" would not match)
# Handles SIDs embedded in strings like "upm_S-1-5-..." (underscore is OK)
sid_pattern = re.compile(r"(?<![A-Za-z])S-1-\d+(?:-\d+)+", re.IGNORECASE)
file_numbers_pattern = re.compile(r"(.+?)\d+")

file_numbers_pattern_2 = re.compile(
    r'\s([A-Za-z_]+[A-Za-z0-9_]*?)\-?(\d+)(?=\s|\'|"|$)'
)

# Git commit hash pattern
# Matches full SHA-1 (40 hex chars) or abbreviated short form (7 hex chars)
# Must contain at least one letter (a-f) to distinguish from pure numbers
# Only replaced when "git" is in the command to avoid false positives
git_commit_hash = re.compile(
    r"\b(?=(?:[0-9]*[a-f])+)[0-9a-f]{7}\b|\b(?=(?:[0-9]*[a-f])+)[0-9a-f]{40}\b",
    re.IGNORECASE,
)

# MD5 hash pattern: exactly 32 hex characters with at least one letter (a-f)
# Uses word boundaries and lookahead to ensure it contains letters (not just digits)
md5_hash_pattern = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{32}\b", re.IGNORECASE)

# Certificate hash pattern: certhash=<40 hex chars (SHA1)>
certhash_pattern = re.compile(
    r"(\b(certhash|code)\s*=\s*)[0-9a-fA-F]{40}\b", re.IGNORECASE
)

some_token_pattern = re.compile(r"\b(?:[A-Za-z0-9]{95,106}|[A-Za-z0-9]{31,33}|[A-Za-z0-9]{128,}|[A-Za-z0-9]{48}|[A-Za-z0-9]{36})\b")
cv_pattern = re.compile(r"[A-Za-z0-9+/]{16,22}(?:\.\d+)+")


# --port patterns with optional quotes around numbers: --port 8080, --port "8080", --port='8080'
port_pattern = re.compile(r'(\s--port\s)["\']?(\d+)["\']?')
port_pattern_2 = re.compile(r'(\s--port=)["\']?(\d+)["\']?')
kill_port_pattern = re.compile(r"\skill-port (\d+)")

docker_port_pattern = re.compile(r'\s+-p\s+\d+:\d+')

port_equals_pattern = re.compile(
    r'(\blistenport\s*=\s*|\bport\s*=\s*|-UniversalPort\s)["\']?(\d+)["\']?',
    re.IGNORECASE,
)

# findstr :port pattern: findstr :59211, findstr ":59211"
findstr_port_pattern = re.compile(r'(findstr\s["\']?:)\d+', re.IGNORECASE)

# Oracle dbms_xplan.display_cursor pattern: display_cursor('sql_id', ...)
dbms_xplan_cursor_pattern = re.compile(
    r"(display_cursor\s*\(\s*')[^']+(')", re.IGNORECASE
)
# Patterns with optional quotes around numbers: port=33331, port="33331", port='33331'
metric_pattern = re.compile(r'(\bmetric\s*=\s*)["\']?(\d+)["\']?', re.IGNORECASE)
interface_pattern = re.compile(r'(\binterface\s*=\s*)["\']?(\d+)["\']?', re.IGNORECASE)
name_equals_pattern = re.compile(
    r'((\bname|/pjob)\s*=\s*)["\']?(\d+)["\']?', re.IGNORECASE
)  # name=10, name="10"

# Sleep pattern: matches sleep preceded by whitespace, " or '
sleep_pattern = re.compile(r'([\s"\'])sleep (\d*[\.\d]\d*)')

# PowerShell -skip, -Seconds, -TimeSpan patterns
skip_pattern = re.compile(r"(-(skip|seconds|timespan)\s)\d+", re.IGNORECASE)

# PowerShell comparison operators with numbers: -eq, -ne, -lt, -gt, -le, -ge
# Examples: Id -ne 14064, -eq 123, -gt 500
ps_compare_pattern = re.compile(r"(-(?:eq|ne|lt|gt|le|ge|id)\s)\d+", re.IGNORECASE)

# PID patterns
pid_pattern = re.compile(r"\b/?pid(?:\seq)?\s(\d+)", re.IGNORECASE)
# PowerShell -id pattern: Get-Process -id 39988
# Matches both ProcessId=1234 and IDProcess=1234, with optional spaces around = or !=, optional quotes
process_id_pattern = re.compile(
    r'\b(ProcessId|IDProcess)\s*!?=?\s*["\']?(\d+)["\']?', re.IGNORECASE
)
# ParentProcessId=14680 → ParentProcessId=<PID>
parent_process_id_pattern = re.compile(
    r'\bParentProcessId\s*!?=?\s*["\']?(\d+)["\']?', re.IGNORECASE
)
event_id_pattern = re.compile(
    r'\b(EventID|IDEvent)\s*!?=?\s*["\']?(\d+)["\']?', re.IGNORECASE
)
# find patterns: find ""1234"", find "1234", find 1234
find_pid = re.compile(r'\bfind\s("{0,2})(\d+)\1(?=\s|$)', re.IGNORECASE)

chcp_pattern = re.compile(r"\bchcp\s(\d+)", re.IGNORECASE)

# Key parameter pattern: --key=value or -key=value or /key=value
# Matches cryptographic keys, API keys, etc. and replaces with <key>
key_param_pattern = re.compile(r'([-/]+key[=:]|SecretKey=)([^\s"]+)', re.IGNORECASE)
apikey_pattern = re.compile(r"(-ApiKey\s)([A-Za-z0-9+/=_-]+)", re.IGNORECASE)

# Token parameter pattern: -Token value or --token value (base64 tokens), with optional quotes
token_param_pattern = re.compile(
    r'((-token|\b[A-Z_]*AUTH[A-Z_]*TOKEN))\s*=?\s*["\']?([\.a-z0-9+/=_-]+)["\']?',
    re.IGNORECASE,
)


# ApiKey parameter pattern: -ApiKey value (base64 or similar keys)

# Auth token environment variables: MCP_PROXY_AUTH_TOKEN=hexvalue, etc.
# PipeName and InstanceName patterns with hex identifiers
pipename_pattern = re.compile(
    r"(-(PipeName|InstanceName)\s)[0-9a-fA-F]+", re.IGNORECASE
)

# Hyphenated hex identifiers:
hyphenated_hex_id_pattern = re.compile(r"\b[0-9a-fA-F]{16,}-[0-9a-fA-F]{16,}\b")

other_key_pattern = re.compile(
    r"\b[a-z0-9]{5}-[a-z0-9]{5}-[a-z0-9]{5}-[a-z0-9]{5}(-[a-z0-9]{5})?"
)

# 64-character hex identifiers (SHA256 or extended IDs)
hex_64_pattern = re.compile(r"\b[0-9A-Fa-f]{64}\b")

# 24-character hex identifiers
hex_24_pattern = re.compile(r"\b[0-9A-Fa-f]{24}\b")

# Client IDs with colon-separated hex segments
client_id_hex_pattern = re.compile(r"\b[A-Z0-9-]+(?::[0-9A-Fa-f]{4}){4,}\b")

# variablePassword pattern: -variablePassword base64value
variable_password_pattern = re.compile(
    r"(-variablePassword\s|Password=|-password\s)['\"]?([A-Za-z0-9+/=]+)['\"]?", re.IGNORECASE
)
username_pattern = re.compile(
    r"(username=|user=)['\"]?([A-Za-z0-9+/=]+)['\"]?", re.IGNORECASE
)

# URL password/session parameters: rmm_session_pwd=123456, session_pwd=abc, pwd=xyz
url_pwd_param_pattern = re.compile(
    r'(\b\w*_?(?:session_)?pwd=)[^\s&"\']+', re.IGNORECASE
)

# acctkey pattern: -acctkey hexvalue
acctkey_pattern = re.compile(r"(-acctkey\s)[0-9a-fA-F]+", re.IGNORECASE)

# AWS ARN pattern: arn:aws:service:region:account-id:resource
aws_arn_pattern = re.compile(r'arn:aws:[a-z0-9-]+::[0-9]+:[^\s"\']+', re.IGNORECASE)

# Encoded params pattern: --encoded-params=<base64> or similar
# Matches base64 encoded parameters and replaces with <encoded>
encoded_params_pattern = re.compile(
    r"([-/]+encoded[-_]?params?[=:])([A-Za-z0-9+/=_-]+)", re.IGNORECASE
)

# AMP file pattern: --amp="something.amp" or --amp=something.amp
# Matches AMP script files and replaces with <amp>
amp_param_pattern = re.compile(r'([-/]+amp[=:])"*([^"]+\.amp)"*', re.IGNORECASE)

# WMIC /node: pattern - matches /node:""hostname"" or /node:"hostname" or /node:hostname
# Used to specify remote computers in WMIC commands
wmic_node_pattern = re.compile(r'/node:"+([^"\s]+)"+|/node:([^\s]+)', re.IGNORECASE)

# WMIC PROCESS pattern - matches "PROCESS 1234" or "process 5678" etc.
# Used to target specific processes by PID in WMIC commands
wmic_process_pattern = re.compile(r"\bprocess\s(\d+)\b", re.IGNORECASE)

# CurDate pattern for wscript: CurDate=DDMMYYYY or CurDate=MMDDYYYY (8 digits)
wscript_curdate_pattern = re.compile(r"\bCurDate=(\d{8})\b", re.IGNORECASE)

# LoadRepNo pattern for wscript: LoadRepNo=""M0024,M0037,..."" or LoadRepNo=value
wscript_loadrepno_pattern = re.compile(
    r'\bLoadRepNo="+([^"]+)"+|\bLoadRepNo=(\S+)', re.IGNORECASE
)

# /try:N pattern for wscript: /try:7 → /try:1
wscript_try_pattern = re.compile(r"/try:(\d+)", re.IGNORECASE)

# -ipk product key pattern: -ipk <key>
# Windows product keys are 5 groups of 5 alphanumeric chars
wscript_ipk_pattern = re.compile(
    r"(-ipk\s)[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}",
    re.IGNORECASE,
)

# WMIC InterfaceIndex pattern - matches "InterfaceIndex=10" etc.
wmic_interface_index_pattern = re.compile(
    r"\bInterfaceIndex\s*=\s*(\d+)", re.IGNORECASE
)

# Duration pattern: /duration followed by number
duration_pattern = re.compile(r"\s/duration (\d+)")

# Standalone filename pattern - matches filename.ext where ext is a script/data extension
# Replaces with just .ext (preserves leading whitespace/boundary)
# Only matches if NOT preceded by / or \ (i.e., not a path)
# Note: <tag>.ext patterns won't match because pattern requires whitespace before filename
standalone_filename_pattern = re.compile(
    r"(?<![/\\])"  # Not preceded by path separator
    r"(\s|^)"  # Capture: whitespace or start of string
    r"[^\s/\\<>]+\."  # Filename: non-whitespace, non-path, non-tag chars, then dot
    r"(py|txt|js|sh|tar|log|dat|bat|ps1|zip|rar|gz|csv|xml|json|yaml|pdf|sln|doc|docx|xls|xlsx|ppt|pptx|cmd)\b",  # Capture: specific extensions
    re.IGNORECASE,
)

# URL pattern - matches http:// and https:// URLs
url_pattern = re.compile(
    r"https?://"  # http:// or https://
    r'[^\s"\'<>]+',  # URL characters (stop at whitespace, quotes, angle brackets)
    re.IGNORECASE,
)

# URL with IP pattern - matches http://<ipv4>/path or http://<ipv6>/path after IP replacement
url_with_ip_pattern = re.compile(
    r"https?://<(int|ext)_ip>"  # http:// or https:// followed by <ipv4> or <ipv6>
    r"(?::\d+)?"  # Optional port
    r'(?:/[^\s"\'<>]*)?',  # Optional path
    re.IGNORECASE,
)

# Domain pattern - matches domain names like example.com, sub.domain.org
# Also captures optional path after domain (e.g., example.com/path/to/page)
# Uses negative lookahead to ensure TLD isn't followed by more letters (e.g., .command)
# Uses negative lookbehind to avoid matching paths like C:\Microsoft.NET
domain_pattern = re.compile(
    r"(?<![/\\])"  # Negative lookbehind: not preceded by path separator
    r"\b[a-z0-9][-a-z0-9]*"  # First word (domain or subdomain)
    r"(?:\.[a-z0-9][-a-z0-9]*)*"  # Zero or more .word segments (subdomains)
    r"\.(com|net|org|io|edu|gov|co|info|biz|click)"  # TLD
    r"(?![a-z\.])"  # Negative lookahead: TLD not followed by letter OR a dot
    r'(?:/[^\s"\'<>]*)?',  # Optional path (e.g., /path/to/page)
    re.IGNORECASE,
)

var_assignment_pattern = re.compile(r'\$([a-zA-Z0-9_:]+)(\s*=)')

# Email pattern - matches email addresses
email_pattern = re.compile(
    r"\b[a-z0-9._%+-]+"  # Local part
    r"@"  # @ symbol
    r"[a-z0-9.-]+"  # Domain
    r"\.[a-z]{2,}\b",  # TLD
    re.IGNORECASE,
)

login_pattern = re.compile(
    r"\b[a-z0-9._%+-]+"  # Local part
    r"@<(ext_ip|int_ip)>",  # @ symbol
    re.IGNORECASE,
)

# Relative path pattern - matches paths starting with ./ or ../ (or .\ or ..\)
# Handles both Unix and Windows style path separators
relative_path_pattern = re.compile(
    r'(?<![^\s"\'<>])'  # Must be preceded by whitespace, quote, or start
    r"\.\.?[/\\]+"  # Start with ./ or ../ (or .\ or ..\)
    r"(?:\.\.?[/\\]+)*"  # Optional additional ../ or ./ (or ..\ or .\)
    r'[^\s"\'<>:?*|]+'  # Rest of the path
)

# Simple relative path pattern - paths with separators but no ./ or ../ prefix
# Matches: dir/file.ext, dir\subdir\file.ext, providers\common\der\file.c.in
# Does NOT match: ./file, ../file, C:\file, /absolute/path, tagged strings (<tag>), domains, env vars
# Similar structure to relative_path_pattern but without ./ or ../ prefix
simple_relative_path = re.compile(
    r"(?<![A-Za-z0-9]\s)"  # Not immediately after word + space (avoid splitting words like "App Name")
    r'(?<![^\s"\'<>%])'  # Must be preceded by whitespace, quote, or start; not after %
    r'(?![^\\s"\'<>]*%[/\\])'  # Do not match paths containing %var% segment
    r'(?![^\\s"\'<>]*:)'  # Do not match segments containing colon (e.g., HKLM:)
    r"(?!<)"  # Do not match already-tagged strings
    r"(?!%)"  # Do not start with environment variable marker
    r"(?![A-Za-z]:)"  # NOT a drive letter
    r"(?![hHkK])"  # NOT a registry key
    r"(?!\.\.?[/\\])"  # NOT starting with ./ or ../
    r"\\?\\?"
    r'[^\s"\'<>:?*|/\\]+'  # First component (no separators, like relative_path_pattern)
    r"[/\\]+"  # At least one separator
    r'[^\s"\'<>:?*|]+'  # Rest of the path (same as relative_path_pattern)
)

# Environment variable path pattern: %VAR%\path\to\file.ext
# We normalize only the path part to <path>\file.ext
env_var_simple_path = re.compile(
    r"%[A-Za-z0-9_]+%"  # %VAR%
    r"[/\\]+"  # separator
    r'[^\s"\'<>]+'  # rest of path
)

# ISO date pattern: YYYYMMDD with optional time, also WMI format YYYYMMDDHHMMSS.microseconds±offset
# Examples: 20251128, 2025-11-28, 20251128000000.000000-000, 2025-12-08T20:07:16Z
date_pattern = re.compile(
    r"20\d{2}[-]?\d{2}[-]?\d{2}(?:[\sT-]?\d{2}[:-]?\d{2}[:-]?\d{2})?(?:\.\d+)?(?:[+-]\d+)?Z?"
)

# US/EU date pattern: MM/DD/YYYY, DD/MM/YYYY, or with dashes, plus optional time
# Examples: 11/27/2025-07:08, 27/11/2025, 31-12-2024, 01-15-2025-14:30:00
slash_date_pattern = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}(?:[-T]\d{2}:\d{2}(?::\d{2})?)?\b"
)

# Time with AM/PM followed by date: "12:54 AM 11/30/2025" → <date>
# Must be matched BEFORE separate time/date patterns
time_ampm_date_pattern = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M\s\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b",
    re.IGNORECASE,
)

# Time with AM/PM pattern (standalone): 12:54 AM, 1:30 PM, 11:59:59 PM
time_ampm_pattern = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M\b", re.IGNORECASE)

# Schtasks /ST time pattern: /ST 21:37, /ST 09:00
schtasks_st_pattern = re.compile(
    r"(/(?:ST|DU)\s)\d{1,2}[:.]\d{2}(?:[:.]\d{2})?",
    re.IGNORECASE
)

# GUID pattern - matches with or without curly braces/quotes, replaces all together
guid_pattern = re.compile(
    r'["\']?\{?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}?["\']?'
)

hives = (
    r"(?:HKEY_LOCAL_MACHINE|HKLM|"
    r"HKEY_CURRENT_USER|HKCU|"
    r"HKEY_CLASSES_ROOT|HKCR|"
    r"HKEY_USERS|HKU|"
    r"HKEY_CURRENT_CONFIG|HKCC)"
)

reg_pattern = re.compile(
    r"(?<!<reg>)"  # Negative lookbehind: not preceded by <reg>
    r"\b(" + hives + r'(?:\\[^"<>|/\r\n]*?)?)'
    r'(?=\s[^a-z0-9]|["\']|$|\))',  # Lookahead: Stop at space, quote, end of line, or closing paren
    re.IGNORECASE,
)

chrome_message_pattern = re.compile(
    r"chrome\.nativeMessaging\.(out|in)\.[a-z0-9]{10,32}"
)
chrome_pipe_pattern = re.compile(r"\\pipe\\chrome")
chrome_extension_pattern = re.compile(r"chrome-extension://[a-z0-9]{32}/")

echo_pass_pattern = re.compile(r'echo\s([\'"]?)[a-z0-9]{12,16}\1', re.IGNORECASE)


def replace_registry_path(match):
    return f"(<reg>{match.group(1)})"


def validate_and_replace_timestamp(match):
    """
    Validate if a number string is a valid Unix timestamp and replace it.

    Unix timestamps (seconds since 1970-01-01):
    - 946684800  = 2000-01-01 00:00:00 UTC
    - 4102444800 = 2100-01-01 00:00:00 UTC

    Args:
        match: regex match object with timestamp string

    Returns:
        '<timestamp>' if valid, original string if not
    """
    ts_str = match.group(1)
    try:
        ts = int(ts_str)

        if len(ts_str) == 13:
            ts //= 1000

        if 946684800 <= ts <= 4102444800:
            return "<timestamp>"
        else:
            return ts_str
    except (ValueError, OverflowError):
        return ts_str


def normalize_filename(filename: str, keep_filename=False):
    """
    Normalize filename: keep only extension if present, otherwise keep filename as-is.
    - file.txt -> .txt
    - file.exe -> file.exe
    - myapp -> myapp
    - script.sh -> .sh
    - file_1234567890.dat -> file_<timestamp>.dat (if valid timestamp)
    """
    if "." in filename:
        extension = filename.rsplit(".", 1)[1]
        if "@" in extension:
            extension = extension.split("@")[0]
        if extension == "exe":
            return filename
        if extension.isdigit():
            return ""
        if len(extension) > 8:
            return ".<ext>"
        return "." + extension

    else:
        if not keep_filename:
            return ""
        if filename.isdigit():
            return "<n>"
        if has_timestamp_pattern.match(filename):
            return "<timestamp>"
        filename = file_numbers_pattern.sub(r"\1", filename)
        return filename


def join_path(*parts):
    return "\\".join(filter(lambda x: len(x) > 0, parts))


def combine_path(base_tag, *parts, keep_midle_path=False, keep_filename=False):
    if len(parts) == 0:
        return base_tag
    parts = list(parts)
    filename = normalize_filename(parts.pop(len(parts) - 1), keep_filename).lower()

    middle = ""
    if keep_midle_path and len(parts) > 0:
        middle = parts.pop(0).lower()
        if len(parts) > 0:
            middle = f"{middle}\\<path>"
    elif len(parts) > 1:
        middle = "<path>"

    return join_path(base_tag, middle, filename)


def replace_hierarchical_path(
    base_tag, path, keep_midle_path=False, keep_filename=False
):
    r"""
    Generic path replacement with hierarchy preservation.

    Format: <base_tag>\FirstDir\<path>\file

    Examples:
    - FirstDir\file.exe -> <base_tag>\FirstDir\.exe
    - FirstDir\sub1\file.exe -> <base_tag>\FirstDir\<path>\.exe
    - FirstDir\sub1\sub2\myapp -> <base_tag>\FirstDir\<path>\myapp
    """
    if path is None:
        return base_tag
    path = path.replace("/", "\\")
    path = re.sub(r"\\+", r"\\", path)
    parts = path.split("\\")

    parts = [p for p in parts if p]

    return combine_path(
        base_tag, *parts, keep_midle_path=keep_midle_path, keep_filename=keep_filename
    )


def replace_program_files(match):
    """Replace Program Files paths."""
    # Four-branch pattern: group(1)=after ", group(2)=after ', group(3)=after space, group(4)=unquoted
    path = match.group(1) or match.group(2) or match.group(3) or match.group(4)
    return replace_hierarchical_path("<program_files>", path)


def replace_program_data(match):
    """Replace ProgramData paths."""
    # Four-branch pattern: group(1)=after ", group(2)=after ', group(3)=after space, group(4)=unquoted
    path = match.group(1) or match.group(2) or match.group(3) or match.group(4)
    return replace_hierarchical_path("<ProgramData>", path, keep_midle_path=True)


def replace_users_path(match):
    """Replace Users paths."""
    # Four-branch pattern: group(1)=after ", group(2)=after ', group(3)=after space, group(4)=unquoted
    path = match.group(1) or match.group(2) or match.group(3) or match.group(4)
    return replace_hierarchical_path("<Users>", path)


def replace_windows_path(match):
    """Replace Windows system paths."""
    # Four-branch pattern: group(1)=after ", group(2)=after ', group(3)=after space, group(4)=unquoted
    path = match.group(1) or match.group(2) or match.group(3) or match.group(4)
    return replace_hierarchical_path("<Windows>", path, keep_filename=True)


def replace_c_drive_generic(match):
    """Replace generic C: drive paths (not covered by specific patterns)."""
    # Three-branch pattern: group(1)=double-quoted, group(2)=single-quoted, group(3)=unquoted
    path = match.group(1) or match.group(2) or match.group(3)
    return replace_hierarchical_path("<C>", path, keep_midle_path=True)


def replace_relative_path(match):
    r"""
    Replace relative paths (./... or ../...) with <path>\.ext or <path>\filename.
    Uses replace_hierarchical_path with just the filename for consistent formatting.
    Examples:
    - ../../db-backup.sh → <path>\.sh
    - ./script.py → <path>\.py
    - ../volume-backups/data.tar.gz → <path>\.gz
    - ../run → <path>\run (no extension)
    """
    path = match.group(0)

    if path.startswith("./"):
        path = path[2:]

    last_slash = max(path.rfind("/"), path.rfind("\\"))
    if last_slash >= 0:
        filename = path[last_slash + 1 :]
        keep_filename = False
    else:
        filename = path
        keep_filename = True

    return replace_hierarchical_path("<path>", filename, keep_filename)


def replace_simple_relative_path(match):
    r"""
    Replace simple relative paths (dir/file.ext) with <path>\.ext or <path>\filename.
    Uses replace_hierarchical_path for consistent formatting (same as replace_relative_path).
    Examples:
    - util/dofile.pl → <path>\.pl
    - providers/common/der/file.c → <path>\.c
    """
    path = match.group(0)

    last_slash = max(path.rfind("/"), path.rfind("\\"))
    if last_slash >= 0:
        filename = path[last_slash + 1 :]
    else:
        filename = path

    return replace_hierarchical_path("<path>", filename)


def replace_env_var_simple(match):
    r"""
    Replace %VAR%\path\to\file with <path>\file (normalize path portion).
    """
    full = match.group(0)
    try:
        second_pct = full.index("%", 1)
    except ValueError:
        return full
    tail = full[second_pct + 1 :]

    last_sep = max(tail.rfind("/"), tail.rfind("\\"))
    filename = tail[last_sep + 1 :] if last_sep >= 0 else tail
    return replace_hierarchical_path("<path>", filename)


def replace_unix_standard_dirs(match):
    """
    Replace standard Unix directories with simplified path.
    Examples:
    - /etc/postinstall/script.sh → <path>\\.sh
    - /usr/bin/bash → <path>\\bash
    - /usr/local/bin/python3 → <path>\\python3
    - /var/log/system.log → <path>\\.log
    """
    subpath = match.group(2)

    if not subpath or subpath == "/":
        base_dir = match.group(1).lower()
        return f"\\{base_dir}"

    subpath = subpath.lstrip("/")
    last_slash = subpath.rfind("/")
    if last_slash >= 0:
        return join_path("<path>", normalize_filename(subpath[last_slash + 1 :]))
    base_dir = match.group(1).lower()
    return f"\\{base_dir}"


def replace_unix_path(match):
    """Replace Unix-like paths on Windows with normalized filename."""
    return join_path("<path>", normalize_filename(match.group(1)))


def replace_windows_other_drives(match):
    """Replace Windows D-Z drive paths with hierarchical normalization."""
    path = match.group(1)  # Everything after "D:\" or "E:\" etc.
    return replace_hierarchical_path("<path>", path)


def replace_unc_ip_path(match):
    r"""
    Replace UNC paths with IP addresses: \\<ipv4>\share\path\file → \\<ipv4>\<path>\file
    """
    path = match.group(1) or match.group(2) or match.group(3) or match.group(4)

    full_match = match.group(0)
    ip_tag = "<int_ip>" if "<int_ip>" in full_match else "<ext_ip>"

    path = path.replace("/", "\\")
    parts = [p for p in path.split("\\") if p]

    return combine_path(ip_tag, *parts)


def replace_unc_host_path(match):
    r"""
    Replace UNC paths with hostnames: \\hostname\share\path\file → \\<host>\<path>\file
    """
    # Four groups: group(1)=quoted hostname, group(2)=quoted path, group(3)=unquoted hostname, group(4)=unquoted path
    path = match.group(2) or match.group(4)

    path = path.replace("/", "\\")
    parts = [p for p in path.split("\\") if p]

    return combine_path(r"<host>", *parts)


def replace_msvc_out(match):
    r"""Replace MSVC /out:path with /out:<path>\.ext"""
    path = match.group(1)
    path = path.replace("/", "\\")
    parts = [p for p in path.split("\\") if p]
    if len(parts) == 0:
        return "/out:"
    filename = normalize_filename(parts[-1])
    if len(parts) == 1:
        return f"/out:{filename}"
    return f"/out:<path>\\{filename}"


def replace_msvc_obj(match):
    r"""Replace MSVC object file paths like dir\file.obj with <path>\.obj"""
    path = match.group(1)
    # Normalize the path
    path = path.replace("/", "\\")
    parts = [p for p in path.split("\\") if p]
    if len(parts) <= 1:
        return normalize_filename(path)
    filename = normalize_filename(parts[-1])
    return f"<path>\\{filename}"


caret_pre_escape = re.compile(r"(\w)\^\&([\s\S])")
caret_escape = re.compile(r"\^([\s\S])")
cmd_whitespace = re.compile(r'("[^"]*")|([,;])')
powershell_escape = re.compile(r"('(?:''|[^'])*')|`([\s\S])")


def separate_flags(match):
    return match.group().replace("/", " /")


def cmd_whitespace_callback(match):
    if match.group(1):
        return match.group(1)
    else:
        return " "


def powershell_escape_callback(match):
    if match.group(1):
        return match.group(1)
    return match.group(2)


def can_be_simple_exe(command):
    command = command.strip('"')
    if not command.endswith(".exe"):
        return False
    if command.count(".") != 1:
        return False
    if not (command.startswith("C:\\") or command.startswith("c:\\")):
        return False
    for char in command[3:]:
        if (
            char
            not in "\\0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ ."
        ):
            return False
    return True

id_pattern = re.compile(r'([-=]|(?<!:):)[A-Za-z0-9]{16}\b')


def process(command):
    if can_be_simple_exe(command):
        to_split = "\\"
        if "\\\\" in command:
            to_split = "\\\\"
        parts = command.split(to_split)
        if " " in parts[-1] and len(parts[-1]) > 10:
            parts[-1] = ".exe"
        if len(parts) > 3:
            parts = parts[:2] + parts[-1:]
        command = to_split.join(parts)
    command = chrome_message_pattern.sub(r"chrome.nativeMessaging.\1.<id>", command)
    command = chrome_pipe_pattern.sub("chrome", command)
    command = chrome_extension_pattern.sub("<chrome_extension_id>", command)
    command = echo_pass_pattern.sub("echo <pass>", command)

    command = hex_64_pattern.sub(r"<sha256>", command)

    for short_pattern, full_path in DOS_SHORTNAME_MAP.items():
        command = re.sub(short_pattern, full_path, command, flags=re.IGNORECASE)

    command = dos_shortname_generic.sub(r"\1", command)

    if "/tmp/" in command:
        command = tmp_files_linux.sub(r"\1<tmp>.\3", command)
        command = tmp_files_linux_no_ext.sub(r"\1<tmp>", command)

    lowercase_command = command.lower()

    if "cmd.exe" in lowercase_command:
        if "^" in command:
            command = caret_pre_escape.sub(r"\1\2", command)
            command = caret_escape.sub(r"\1", command)
        if ";" in command or "," in command:
            command = cmd_whitespace.sub(cmd_whitespace_callback, command)

    if "powershell.exe" in lowercase_command and "`" in command:
        command = powershell_escape.sub(powershell_escape_callback, command)

    if "temp" in lowercase_command:
        command = c_temp_ext.sub(r"<tmp>.\1", command)
        command = c_temp_no_ext.sub(r"<tmp>", command)
        command = windows_temp_ext.sub(r"<tmp>.\1", command)
        command = windows_temp_no_ext.sub(r"<tmp>", command)
        command = users_temp_ext.sub(r"<tmp>.\1", command)
        command = users_temp_no_ext.sub(r"<tmp>", command)

    command = flag_cluster_pattern.sub(separate_flags, command)

    if "hk" in command:
        command = reg_pattern.sub(replace_registry_path, command)

    has_slash = "/" in command
    if has_slash:
        command = unix_standard_dirs.sub(replace_unix_standard_dirs, command)
    command = program_files.sub(replace_program_files, command)
    command = program_data.sub(replace_program_data, command)
    command = users_path.sub(replace_users_path, command)
    command = windows_path.sub(replace_windows_path, command)
    command = c_drive_generic.sub(replace_c_drive_generic, command)
    command = windows_other_drives.sub(replace_windows_other_drives, command)

    if (
        "msbuild" in lowercase_command
        or "msvc" in lowercase_command
        or "vcxproj" in lowercase_command
    ):
        command = vcxproj_files.sub(r"<vcxproj>", command)
        command = vs_version.sub(r"<vs_version>", command)
        command = vs_configuration.sub(r"<vs_configuration>", command)
        command = vs_platform.sub(r"<vs_platform>", command)
        command = msvc_out_pattern.sub(replace_msvc_out, command)
        command = msvc_obj_pattern.sub(replace_msvc_obj, command)

    command = cv_pattern.sub(r"<id>", command)

    if has_slash:
        command = unix_path_on_windows.sub(replace_unix_path, command)
    command = ipv4_internal_pattern.sub(r"<int_ip>", command)
    command = ipv4_external_pattern.sub(r"<ext_ip>", command)
    command = mac_address_pattern.sub(r"<mac>", command)
    command = ipv6_internal_pattern.sub(r"<int_ip>", command)
    command = ipv6_external_pattern.sub(r"<ext_ip>", command)
    command = login_pattern.sub(r"<user>@<\1>", command)


    command = ssh_pattern.sub(r".ssh\1<ssh>", command)

    command = unc_host_path.sub(replace_unc_host_path, command)
    if "_ip" in command:
        command = unc_ip_path.sub(replace_unc_ip_path, command)
        command = ip_port_pattern2.sub(r"<port>:<\2_ip>:<port>", command)
        command = ip_port_pattern.sub(r"<\1_ip>:<port>", command)
        command = ip_comma_port_pattern.sub(r"<\1_ip>,<port>", command)

    command = tcp_host_port_pattern.sub(r"\1<host>,<port>", command)


    command = docker_port_pattern.sub(r" -p <port>:<port>", command)

    command = env_var_pattern.sub("getenvironmentvariable('<env>')", command)

    if "git" in lowercase_command:
        command = command.replace("0" * 40, "<commit_hash>")
        command = git_commit_hash.sub(r"<commit_hash>", command)
    if " --port" in command:
        command = port_pattern.sub(r"\1<port>", command)
        command = port_pattern_2.sub(r"\1<port>", command)
    if " kill-port " in command:
        command = kill_port_pattern.sub(r" kill-port <port>", command)
    if "findstr" in lowercase_command:
        command = findstr_port_pattern.sub(r"\1<n>", command)
    if "display_cursor" in lowercase_command:
        command = dbms_xplan_cursor_pattern.sub(r"\1<id>\2", command)

    command = port_equals_pattern.sub(r"\1<port>", command)

    command = metric_pattern.sub(r"\1<n>", command)
    command = interface_pattern.sub(r"\1<n>", command)

    command = name_equals_pattern.sub(r"\1<n>", command)
    command = sleep_pattern.sub(r"\1sleep <s>", command)

    command = skip_pattern.sub(r"\1<n>", command)
    command = pid_pattern.sub(r"pid <PID>", command)

    command = ps_compare_pattern.sub(r"\1<PID>", command)

    command = command.replace('"pid <PID>"', "pid <PID>")

    command = process_id_pattern.sub(r"ProcessId <PID>", command)
    command = parent_process_id_pattern.sub(r"ParentProcessId <PID>", command)
    command = event_id_pattern.sub(r"EventID <id>", command)

    command = find_pid.sub(r"find <PID>", command)

    command = chcp_pattern.sub(r"chcp <code>", command)

    command = apikey_pattern.sub(r"\1<key>", command)
    command = key_param_pattern.sub(r"\1<key>", command)

    command = token_param_pattern.sub(r"\1<token>", command)

    command = pipename_pattern.sub(r"\1<id>", command)
    command = id_pattern.sub(r"\1<id>", command)

    command = hyphenated_hex_id_pattern.sub(r"<id>", command)
    command = other_key_pattern.sub(r"<key>", command)
    command = hex_24_pattern.sub(r"<hex_id>", command)
    command = client_id_hex_pattern.sub(r"<client_id>", command)

    command = variable_password_pattern.sub(r"\1<pass>", command)
    command = username_pattern.sub(r"\1<user>", command)
    command = url_pwd_param_pattern.sub(r"\1<pass>", command)

    command = acctkey_pattern.sub(r"\1<key>", command)
    command = aws_arn_pattern.sub(r"<arn>", command)
    command = encoded_params_pattern.sub(r"\1<payload>", command)
    command = amp_param_pattern.sub(r"\1<amp>", command)
    command = wmic_node_pattern.sub(r"/node:<node>", command)

    if "wmic" in lowercase_command:
        command = wmic_process_pattern.sub(r"process <PID>", command)
        command = wmic_interface_index_pattern.sub(r"InterfaceIndex=<id>", command)

    if "wscript" in lowercase_command or "vbs" in lowercase_command:
        command = wscript_curdate_pattern.sub(r"CurDate=<date>", command)
        command = wscript_loadrepno_pattern.sub(r"LoadRepNo=<ids>", command)
        command = wscript_try_pattern.sub(r"/try:1", command)  # /try:7 → /try:1
        command = wscript_ipk_pattern.sub(r"\1<key>", command)

    if " /duration " in command:
        command = duration_pattern.sub(r" /duration <s>", command)

    command = standalone_filename_pattern.sub(r"\1.\2", command)

    command = guid_pattern.sub(r"<guid>", command)
    command = certhash_pattern.sub(r"\1<hash>", command)
    command = md5_hash_pattern.sub(r"<md5>", command)

    command = some_token_pattern.sub(r"<token>", command)

    command = var_assignment_pattern.sub(r"$<env>\2", command)
    command = email_pattern.sub(r"<email>", command)

    command = url_pattern.sub(r"<url>", command)
    command = url_with_ip_pattern.sub(r"<url>", command)
    command = domain_pattern.sub(r"<url>", command)

    command = url_port_pattern.sub(r"<url>:<port>", command)

    command = env_var_simple_path.sub(replace_env_var_simple, command)

    command = relative_path_pattern.sub(replace_relative_path, command)
    command = simple_relative_path.sub(replace_simple_relative_path, command)
    command = file_numbers_pattern_2.sub(r" \1", command)

    command = sid_pattern.sub(r"<SID>", command)


    command = has_timestamp_pattern.sub(r"<timestamp>", command)

    command = time_ampm_date_pattern.sub(r"<date>", command)
    command = date_pattern.sub(r"<date>", command)
    command = slash_date_pattern.sub(r"<date>", command)
    command = time_ampm_pattern.sub(r"<date>", command)

    command = schtasks_st_pattern.sub(r"\1<date>", command)

    command = from_base64_pattern.sub(replace_from_base64, command)

    return command.strip()


four_consecutive_quotes = re.compile(r'"{4,}')
double_quotes_word = re.compile(r'""([^\s"]+)""')
space_before_tag = re.compile(r"<(?![a-zA-Z0-9_]+>|<|\s)")
special_redirect_pattern = re.compile(r"(\d*>&(?:-|\d+)|&>>?)")

multiple_spaces = re.compile(r"\s{2,}")

operators = [
    r"2>&1",
    r"1>&2",  # Stream merging
    r"\|\|",
    r"&&",  # Logical OR/AND
    r"(?<!<[a-zA-Z0-9_]+)>>",
    r"<<",  # Append / Heredoc
    r"&>",  # Bash shortcut
    r"\|",  # Pipe
]

operators_pattern = re.compile(r"(" + "|".join(operators) + r")")


def preprocess(command):
    command = command.strip()
    command = operators_pattern.sub(r" \1 ", command)
    command = multiple_spaces.sub(" ", command)

    lead_quotes = 0
    for char in command:
        if char == '"':
            lead_quotes += 1
        else:
            break

    trail_quotes = 0
    for char in reversed(command):
        if char == '"':
            trail_quotes += 1
        else:
            break

    if (lead_quotes >= 3 and trail_quotes >= 1) or (
        lead_quotes >= 1 and trail_quotes >= 3
    ):
        command = command[1:-1]

    command = double_quotes_word.sub(r'"\1"', command)

    command = space_before_tag.sub(r"< ", command)

    command = special_redirect_pattern.sub(r"<r>", command)

    return command.strip()

def safe_base64_decode(encoded_str):
    """
    Adds missing padding and attempts to decode Base64 string.
    Returns bytes or None.
    """
    if not encoded_str:
        return None

    # Fix padding: length must be divisible by 4
    pad = len(encoded_str) % 4
    if pad > 0:
        encoded_str += '=' * (4 - pad)

    try:
        return base64.b64decode(encoded_str)
    except Exception:
        return None


def get_byte_candidates(raw_bytes):
    """
    Returns a list of byte arrays to try decoding.
    Checks for raw bytes AND compressed (Deflate) payloads.
    """
    candidates = [raw_bytes]

    # Check for PowerShell compression (DeflateStream)
    # zlib.decompress(data, -15) handles raw deflate (no headers)
    try:
        decompressed = zlib.decompress(raw_bytes, -15)
        candidates.append(decompressed)
    except Exception:
        pass
    try:
        decompressed = gzip.decompress(raw_bytes)
        candidates.append(decompressed)
    except Exception:
        pass
    return candidates

def safe_text_decode(byte_data, encoding):
    """
    Decodes bytes to text, handling truncated multi-byte characters.
    Returns string or None.
    """
    try:
        # Handle Truncated UTF-16 (PowerShell's default)
        # UTF-16 requires 2 bytes per char. If we have an odd number of bytes
        # (due to log truncation), we must drop the last byte to decode successfully.
        data_to_decode = byte_data
        if encoding.startswith("utf-16") and len(byte_data) % 2 != 0:
            data_to_decode = byte_data[:-1]

        return data_to_decode.decode(encoding)
    except Exception:
        return None


def calculate_readability_score(text):
    """
    Returns a score (0.0 to 1.0) representing how 'readable' the text is.
    """
    if not text:
        return 0.0

    total_len = len(text)
    if total_len == 0:
        return 0.0

    # We count printable characters plus standard whitespace (newlines, tabs)
    # We treat null bytes (\x00) as non-printable garbage
    valid_chars = sum(1 for c in text if c.isprintable() or c in "\r\n\t")

    return valid_chars / total_len

control_chars = re.compile(r'[\x01-\x08\x0b\x0c\x0e-\x1f]')

def is_binary_garbage(text):
    """
    Checks if a string contains non-printable binary control characters.
    Ignores Null (\x00), Tab (\x09), Newline (\x0a), and Carriage Return (\x0d).
    """
    # Regex for ASCII control characters (01-08, 0B-0C, 0E-1F)
    # If these exist in high numbers, it is compressed binary, NOT a script.
    matches = control_chars.findall(text)

    # If even 2% of the string is control characters, it's binary data
    if len(matches) / len(text) > 0.02:
        return True

    return False

def try_decode_powershell_encoded(encoded_str):
    """
    Main Orchestrator Function.
    1. Fixes Base64 padding.
    2. Checks for compression.
    3. Tries multiple encodings (UTF-16LE, UTF-8, CP1252).
    4. Returns the result with the highest readability score.
    """
    # 1. Decode Base64
    raw_bytes = safe_base64_decode(encoded_str)
    if not raw_bytes:
        return None

    # 2. Prepare candidates (Raw bytes vs Compressed bytes)
    byte_candidates = get_byte_candidates(raw_bytes)

    # 3. Define Encodings to try
    # PowerShell standard is utf-16-le. Web is utf-8. Legacy Windows is cp1252.
    encodings = ["utf-16-le", "utf-8", "cp1252", "latin-1"]

    best_text = None
    best_score = 0.0

    # 4. Decode and Score
    for bytes_data in byte_candidates:
        for encoding in encodings:
            decoded_text = safe_text_decode(bytes_data, encoding)

            if decoded_text:
                if is_binary_garbage(decoded_text):
                    continue
                score = calculate_readability_score(decoded_text)

                # We want the highest score.
                # If scores are tied, we stick with the first one found
                # (which respects our encoding priority list).
                if score > best_score:
                    best_score = score
                    best_text = decoded_text

    # 5. Final Threshold Check
    # If the best we found is less than 50% readable, it's likely binary garbage.
    if best_score < 0.5:
        return None

    # '謟' (\u8b1f) is exactly the GZIP header (1F 8B) interpreted as UTF-16LE
    bad_chars = "謟\u8b1f␊獡摤獦晤㐱㴠∠"
    if any(x in best_text for x in bad_chars):
        return None

    # Clean up Null bytes which are common artifacts in Windows logs
    return best_text.replace('\x00', '')


def try_decode_base64_utf8(encoded_str):
    """
    Try to decode a regular Base64 UTF-8 string (used in FromBase64String).

    Returns:
        (True, decoded_string) if successful and produces readable text
        (False, None) if decoding fails or produces garbage
    """
    decoded = try_decode_powershell_encoded(encoded_str)

    try:
        decoded_bytes = base64.b64decode(encoded_str)
        decoded = decoded_bytes.decode("utf-8")

        # Check if result looks like readable text
        printable_ratio = (
            sum(1 for c in decoded if c.isprintable() or c in "\r\n\t") / len(decoded)
            if decoded
            else 0
        )
        if printable_ratio > 0.8:
            return (True, decoded)
        else:
            return (False, None)
    except Exception:
        return (False, None)


from_base64_pattern = re.compile(
    r"(?:FromBase64String|base64\.b64decode)"  # Match either function name
    r"\s*\(\s*"                                 # Open parenthesis and whitespace
    r"(?:[bB]?['\"\\]*)?"                       # Optional opening quotes (handles ', ", \", b', etc.)
    r"([A-Za-z0-9+/=]+)"                        # CAPTURE GROUP 1: The Base64 string
    r"(?:['\"\\]*)?"                            # Optional closing quotes
    r"\s*\)",                                   # Close parenthesis
    re.IGNORECASE
)

def replace_from_base64(match):
    """Replace FromBase64String('...') with decoded content or <encoded>"""
    encoded = match.group(1)
    decoded = try_decode_powershell_encoded(encoded)
    if decoded is not None:
        decoded_escaped = decoded.replace("'", "''")
        return f"{decoded_escaped}"
    else:
        return "<payload>"


ENCODED_COMMAND_INDICATORS = {
    "-encodedcommand",
    "-encodedcomman",
    "-encodedcomma",
    "-encoded",
    "-encodedcomm",
    "-encodedco",
    "-encodedc",
    "-enc",
    "-ec",
    "-encodedarguments",
    "-encodedargument",
    "-ea",
    "-paramsasbase64",
    "/encodedcommand",
    "/encodedcomman",
    "/encodedcomma",
    "/encoded",
    "/encodedcomm",
    "/encodedco",
    "/encodedc",
    "/enc",
    "/ec",
    "/encodedarguments",
    "/encodedargument",
    "/ea",
    "/paramsasbase64"
}

env_var_pattern = re.compile(
    r"getenvironmentvariable\s*\(\s*['\"][^'\"]+['\"]\s*\)",
    re.IGNORECASE
)


filename_pattern_valid = re.compile(r"^[a-zA-Z0-9]*(?:\s?\(\d+\))*$")

def is_valid_filename_format(text):
    return bool(filename_pattern_valid.match(text))




def cleanup_token(token: str):
    r"""
    Post-tokenization cleanup for tokens that start with normalized tags
    but still have unmatched path elements, AND simple relative paths.

    Examples:
        <tmp>.NET Files\admin.vswh\10710812\file.cmdline → <tmp>.cmdline
        @""<tmp>.NET Files\...\file.cmdline → @<tmp>.cmdline
        <path>\subdir\subdir2\file.exe → <path>\file.exe (keeps exe name)
        temp\patches\scripts123\script.ps1 → <path>\.ps1
    """
    if token.count(".") == 1:
        part_1, part_2 = token.split(".")
        if "exe" not in part_2 and is_valid_filename_format(part_1) and " " not in part_2 and all(x not in part_2 for x in path_forbidden_chars):
            return "." + part_2

    token = token.lstrip('\\')

    if token.isdigit():
        return "<n>"
    if token.strip("()").isdigit():
        return "<n>"
    for start in "-:=":
        if token.startswith(start) and token[1:].isdigit():
            return start, "<n>"
        if token.count(start) == 1:
            part_1, part_2 = token.split(start)
            if part_2.isdigit():
                return part_1, "<n>"

    # Handle prefix like @"" or @
    prefix = ""
    work = token
    if work.startswith("@"):
        prefix = "@"
        work = work[1:].strip('"').lstrip('\\')

    # Check if token starts with a tag like <tmp>, <path>, <Users>, etc.
    tag_match = re.match(r"^(<[a-zA-Z_-]+>)", work)
    if tag_match:
        tag = tag_match.group(1)
        rest = work[len(tag) :]

        # If this is a <tmp> token with extra path elements, simplify to <tmp>.ext
        if (
            tag in ("<tmp>", "<path>", "<Users>", "<ProgramFiles>", "<Windows>")
            and rest
        ):
            # Find the last extension
            pos = rest.rfind(".")
            if pos >= 0:
                if rest[pos + 1 :] == "exe":
                    return work
                if rest == "\\<path>\\." + rest[pos + 1 :]:
                    return work
                if tag == "<tmp>":
                    return prefix + tag + "." + rest[pos + 1 :]
                else:
                    return prefix + tag + "\\." + rest[pos + 1 :]
            else:
                return work

        return token

    # Check for simple relative paths (no drive letter, no leading / or \, has separators)
    # This handles paths like: temp\patches\script.ps1, dir/subdir/file.txt
    # Only match if the token looks like JUST a path (no spaces, no special chars except path separators)
    if not work.startswith(("<", "/", "\\", "%", "-", "(", ")")) and not re.search(
        r"(^[A-Za-z]:|[hHkK])", work
    ):
        # Check if it looks like a pure path (has separator, no spaces, no colons, no quotes)
        if (
            ("\\" in work or "/" in work)
            and " " not in work
            and ":" not in work
            and '"' not in work
            and '\'' not in work
            and '?' not in work
        ):
            return prefix + replace_simple_relative_path_token(work)

    return token


def replace_simple_relative_path_token(path):
    r"""
    Normalize a simple relative path token to <path>\.ext format.
    Called from cleanup_token for paths like temp\patches\script.ps1
    """
    last_slash = max(path.rfind("/"), path.rfind("\\"))
    if last_slash >= 0:
        filename = path[last_slash + 1 :]
    else:
        filename = path

    return replace_hierarchical_path("<path>", filename)

path_forbidden_chars = "|*?:"

def is_not_path(token, prev_command=None):
    # print("token:       ", token, len(token))
    # print("prev_command:", prev_command, len(prev_command) if prev_command else 0)
    # import time
    # time.sleep(1)
    if token == prev_command:
        return False
    if token.count(".") == 1:
        part_1, part_2 = token.split(".")
        if is_valid_filename_format(part_1) and " " not in part_2 and all(x not in part_2 for x in path_forbidden_chars):
            return False
    if any(char in token for char in path_forbidden_chars):
        return True

    if "/" not in token and "\\" not in token and " " in token:
        return True

    if ".exe" in token and not token.endswith(".exe"):
        return True

    if " <" in token:
        return True
    if '"' in token.strip('"'):
        return True
    if "'" in token.strip("'"):
        return True
    if token.count("<") != token.count(">"):
        return True
    # TODO: This statistics can be refined
    if len(token) > 20 and token.count(" ") > 4:
        return True
    if len(token) > 40 and token.count(" ") > 3:
        return True
    if " -e" in token.lower() and token.count(" ") > 2 and len(token) > 40:
        return True
    return False


def python_windows_split_non_regex(cmdline):
    """
    Splits a string into command line arguments using MSVCRT (Windows) rules.
    This mimics the logic of CommandLineToArgvW without using ctypes.
    """
    args = []
    current_arg = []
    backslashes = 0
    in_quotes = False

    # We need to know if we are currently building an argument
    # to handle the difference between "whitespace between args" and "empty arg"
    arg_started = False

    for i, c in enumerate(cmdline):
        if c == '\\':
            backslashes += 1
            arg_started = True
            continue

        if c == '"':
            arg_started = True
            # Even number of backslashes means they are literal backslashes,
            # and the quote is a delimiter (starts or ends a string).
            if backslashes % 2 == 0:
                current_arg.append('\\' * (backslashes // 2))
                in_quotes = not in_quotes
            # Odd number of backslashes means they are literal backslashes,
            # followed by an escaped (literal) quote.
            else:
                current_arg.append('\\' * (backslashes // 2))
                current_arg.append('"')
            backslashes = 0
            continue

        # If we have pending backslashes followed by a normal character,
        # they are just literal backslashes.
        if backslashes > 0:
            current_arg.append('\\' * backslashes)
            backslashes = 0

        # Handle whitespace
        if c.isspace() and not in_quotes:
            if arg_started:
                args.append("".join(current_arg))
                current_arg = []
                arg_started = False
            # If not arg_started, we just ignore the whitespace
        else:
            current_arg.append(c)
            arg_started = True

    # Handle the last argument if exists
    if arg_started:
        if backslashes > 0:
            current_arg.append('\\' * backslashes)
        args.append("".join(current_arg))

    return args

def python_windows_split(cmdline):
    # Regex to match:
    # 1. Quoted strings (handling escaped quotes like \")
    # 2. Non-whitespace sequences
    # Note: This is an approximation. Windows parsing is actually not regex-able
    # in 100% of cases due to state, but this covers 99%.
    pattern = r'"((?:[^"]|\\")*)"|(\S+)'

    tokens = []
    for quoted, plain in re.findall(pattern, cmdline):
        if quoted:
            # Windows un-escaping: replace \" with "
            tokens.append(quoted.replace('\\"', '"'))
        else:
            tokens.append(plain)
    return tokens


import ctypes

if os.name == "nt":
    from ctypes import wintypes

    _CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
    _CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    _CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]

    _LocalFree = ctypes.windll.kernel32.LocalFree
    _LocalFree.argtypes = [wintypes.HLOCAL]
    _LocalFree.restype = wintypes.HLOCAL


def win_split(cmdline):
    argc = ctypes.c_int()
    argv = _CommandLineToArgvW(cmdline, ctypes.byref(argc))

    if not argv:
        return []

    try:
        return [argv[i] for i in range(argc.value)]
    finally:
        _LocalFree(argv)


class WindowsFallbackWarning(UserWarning):
    pass

# 2. Configure the filter to ONLY show this warning once
warnings.simplefilter('once', WindowsFallbackWarning)


BASE64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')
delimiter = r'[-:=;\.\)\( \'"\s\\\/_,\[\]\{\}&\|]'
cleanup_numbers = re.compile(
    fr'(?:^|(?<={delimiter}))\d+(?={delimiter}|$)|\d+$|^\d+'
)


long_chars = re.compile(
    r"[-a-z0-9\.=]{60,}"
)


def post_clean_token(token: str):
    while "\\\\" in token:
        token = token.replace("\\\\", "\\")
    while "\"\"" in token:
        token = token.replace("\"\"", "\"")
    token = cleanup_numbers.sub("<n>", token)
    token = token.lower()
    token = long_chars.sub("<token>", token)
    token = token.replace("<token><n>", "<token>")
    if token.startswith("$") and token.replace("<", "").replace(">", "").replace(":", "")[1:].isalnum():
        return "$<env>"
    if " " not in token:
        maybe_token_indicators = ("<token>", "<date>", "<n>")
        if any([x in token and x != token for x in maybe_token_indicators]) and token.replace("<", "").replace(">", "").isalnum():
            return "<token>"
    return token


def split_into_tokens(command):
    if os.name == "nt":
        return win_split(command)
    else:
        warnings.warn("Not on Windows: falling back to regex parser.", WindowsFallbackWarning)
        return python_windows_split(command)

def parse_windows_cmdline(command, prev_command=None, no_normalize=False):
    """
    Parse Windows command line with nested quotes.

    Splits by spaces while respecting quoted sections (both single and double quotes).
    Quotes are stripped from the output.

    Example:
        Input:  '"C:\\cmd.exe" /C " find "x" "'
        Output: ['C:\\cmd.exe', '/C', ' find "x" ']
    """
    if no_normalize:
        if command == prev_command:
            return command.lower()
        tokens = split_into_tokens(command)
        if len(tokens) == 1:
            return tokens[0]
        
        decoded_tokens = []
        prev_token_lower = None
        for token in tokens:
            if not token:
                continue
            if prev_token_lower in ENCODED_COMMAND_INDICATORS or prev_token_lower in ("-e", "/e") and BASE64_RE.match(token):
                decoded = try_decode_powershell_encoded(token)
                if decoded is not None:
                    decoded_tokens.append(decoded)
                else:
                    decoded_tokens.append(token)
            else:
                decoded_tokens.append(token)
            prev_token_lower = token.lower()
        
        return tuple([parse_windows_cmdline(token, command, True) for token in decoded_tokens])
    command = preprocess(command)
    command = process(command)

    tokens = split_into_tokens(command)

    cleaned_tokens = []
    prev_token_lower = ""

    for token in tokens:
        if not token:
            continue

        lower_token = token.lower()
        find_index = lower_token.find("frombase64string")
        if find_index != -1 and find_index + len("frombase64string") < len(token) - 1:
            start = find_index + len("frombase64string")
            preencoded = token[:start]
            parsed_preencoded = parse_windows_cmdline(preencoded, no_normalize=no_normalize)

            encoded = token[start:].strip("(").strip("\"'\\")
            decoded = try_decode_powershell_encoded(encoded)

            if decoded is not None:
                processed_decoded = tuple(parse_windows_cmdline(decoded, no_normalize=no_normalize))
                parsed_preencoded.append(processed_decoded)
            else:
                parsed_preencoded.append("<payload>")

            cleaned_tokens.append(tuple(parsed_preencoded))
        elif prev_token_lower in ENCODED_COMMAND_INDICATORS or prev_token_lower in ("-e", "/e") and BASE64_RE.match(token):
            decoded = try_decode_powershell_encoded(token)
            if decoded is not None:
                processed_decoded = tuple(parse_windows_cmdline(decoded, no_normalize=no_normalize))
                cleaned_tokens.append(processed_decoded)
            else:
                cleaned_tokens.append("<payload>")
        else:
            if is_not_path(token, prev_command):
                tokens = parse_windows_cmdline(token, token, no_normalize=no_normalize)
                # Was post cleaned
                if len(tokens) > 1:
                    cleaned_tokens.append(tuple(tokens))
                elif len(tokens) == 1:
                    cleaned_tokens.append(tokens[0])
            else:
                token = token.strip("\"'").strip()
                token = cleanup_token(token)
                if isinstance(token, str):
                    cleaned_tokens.append(post_clean_token(token))
                else:
                    cleaned_tokens.extend([post_clean_token(x) for x in token])

        prev_token_lower = token.lower() if isinstance(token, str) else None

    return cleaned_tokens


if __name__ == "__main__":
    import sys
    no_normalize = False
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test.txt":
            with open("test.txt", "r") as f:
                command = f.read()
            if len(sys.argv) > 2:
                no_normalize = sys.argv[2] == "True"
        else:
            command = " ".join(sys.argv[1:])

        print("ORIGINAL COMMAND:")
        print(command)
        print()

        # Process
        preprocessed = preprocess(command)
        print("PREPROCESSED COMMAND:")
        print(preprocessed)
        print()
        processed = process(preprocessed)
        print("PROCESSED COMMAND:")
        print(processed)
        print()

        try:
            tokens = parse_windows_cmdline(command, no_normalize=no_normalize)
            print(f"TOKENS ({len(tokens)}):")
            for idx, token in enumerate(tokens):
                print(f"  [{idx}]: {token}")
                if isinstance(token, tuple):
                    for t in token:
                        print(f"            -> {t}")

            print()
        except Exception as e:
            print(f"ERROR: {e}")
            print("=" * 80)
            print("✗ Parse failed")
            print("=" * 80)
    else:
        print("Usage: python tokenize.py <command>")
