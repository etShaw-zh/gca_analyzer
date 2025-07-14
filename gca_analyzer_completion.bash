#!/bin/bash
# Bash completion for gca_analyzer

_gca_analyzer_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="--help -h --interactive -i --data --output --best-window-indices --act-participant-indices --min-window-size --max-window-size --model-name --model-mirror --default-figsize --heatmap-figsize --log-file --console-level --file-level --log-rotation --log-compression"
    
    case ${prev} in
        --data)
            # Complete with CSV files
            COMPREPLY=( $(compgen -f -X "!*.csv" -- ${cur}) )
            return 0
            ;;
        --output)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- ${cur}) )
            return 0
            ;;
        --console-level|--file-level)
            # Complete with log levels
            COMPREPLY=( $(compgen -W "DEBUG INFO WARNING ERROR CRITICAL" -- ${cur}) )
            return 0
            ;;
        --log-file)
            # Complete with log files
            COMPREPLY=( $(compgen -f -X "!*.log" -- ${cur}) )
            return 0
            ;;
        --model-name)
            # Complete with common model names
            COMPREPLY=( $(compgen -W "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers/all-MiniLM-L6-v2" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

# Register the completion function
complete -F _gca_analyzer_completion gca_analyzer
complete -F _gca_analyzer_completion python -m gca_analyzer