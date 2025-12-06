#!/usr/bin/env python3
"""
Rate limit status monitor for RED_CORE.

Shows current rate limit status for all configured providers.
"""

import time
import argparse
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from app.core.rate_limiter import get_rate_limiter


def format_tokens(tokens: int) -> str:
    """Format token count with K/M suffixes."""
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.1f}K"
    else:
        return str(tokens)


def create_status_table() -> Table:
    """Create a Rich table showing rate limit status."""
    limiter = get_rate_limiter()
    all_status = limiter.get_all_status()
    
    table = Table(title="⚡ Rate Limit Status", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan", width=12)
    table.add_column("Strategy", style="yellow")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Requests", justify="right", style="blue")
    table.add_column("Status", style="bold")
    
    for provider, status in sorted(all_status.items()):
        if not status['configured']:
            continue
            
        # Format token status
        tokens_avail = format_tokens(status['available_tokens'])
        tokens_cap = format_tokens(status['token_capacity'])
        token_pct = (status['available_tokens'] / status['token_capacity'] * 100) if status['token_capacity'] > 0 else 0
        token_str = f"{tokens_avail}/{tokens_cap} ({token_pct:.0f}%)"
        
        # Format request status
        req_pct = (status['available_requests'] / status['request_capacity'] * 100) if status['request_capacity'] > 0 else 0
        req_str = f"{status['available_requests']}/{status['request_capacity']} ({req_pct:.0f}%)"
        
        # Status indicator
        if status['in_backoff']:
            wait_time = status['backoff_until'] - time.time()
            status_str = f"[red]BACKOFF ({wait_time:.0f}s)[/red]"
        elif token_pct < 20 or req_pct < 20:
            status_str = "[yellow]LOW[/yellow]"
        else:
            status_str = "[green]OK[/green]"
        
        # Color tokens/requests based on availability
        if token_pct < 20:
            token_str = f"[red]{token_str}[/red]"
        elif token_pct < 50:
            token_str = f"[yellow]{token_str}[/yellow]"
            
        if req_pct < 20:
            req_str = f"[red]{req_str}[/red]"
        elif req_pct < 50:
            req_str = f"[yellow]{req_str}[/yellow]"
        
        table.add_row(
            provider.upper(),
            status['strategy'],
            token_str,
            req_str,
            status_str
        )
    
    return table


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor rate limit status")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor status")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds (default: 5)")
    args = parser.parse_args()
    
    console = Console()
    
    if args.watch:
        # Live monitoring mode
        console.print("[bold cyan]⚡ Rate Limit Monitor[/bold cyan]")
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")
        
        try:
            with Live(create_status_table(), console=console, refresh_per_second=1) as live:
                while True:
                    time.sleep(args.interval)
                    live.update(create_status_table())
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped[/yellow]")
    else:
        # Single status display
        console.print(create_status_table())
        
        # Show tips
        tips = """
[dim]Tips:[/dim]
• Set environment variables to override limits (e.g., OPENAI_RATE_LIMIT_TPM=3000000)
• Use --watch flag for continuous monitoring
• Providers with 'pre_emptive' strategy will wait before hitting limits
• Providers with 'reactive' strategy will retry after hitting limits
"""
        console.print(Panel(tips, title="[bold blue]💡 Rate Limiting Tips[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    main()