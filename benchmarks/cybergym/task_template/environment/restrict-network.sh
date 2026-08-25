#!/usr/bin/env bash
# Runtime egress policy for the agent container. Docker builds still have
# network access, while evaluated agents default to private Docker networks
# only. Set CYBERGYM_NETWORK_MODE=public to opt out, or ALLOWED_HOSTS to a
# comma-separated allowlist when a controlled external service is required.
set -euo pipefail

NETWORK_MODE="${CYBERGYM_NETWORK_MODE:-restricted}"
REQUIRE_RESTRICTION="${CYBERGYM_REQUIRE_NETWORK_RESTRICTION:-true}"

if [ "$NETWORK_MODE" = "public" ]; then
    echo "cybergym-network: public runtime egress enabled"
    exec "$@"
fi
if [ "$NETWORK_MODE" != "restricted" ]; then
    echo "cybergym-network: invalid CYBERGYM_NETWORK_MODE=$NETWORK_MODE" >&2
    exit 1
fi

if ! iptables -L OUTPUT -n >/dev/null 2>&1; then
    if [ "$REQUIRE_RESTRICTION" = "true" ]; then
        echo "cybergym-network: iptables unavailable; refusing to run without the requested restriction" >&2
        exit 1
    fi
    echo "cybergym-network: WARNING iptables unavailable; runtime egress is not restricted" >&2
    exec "$@"
fi

ALLOWED_IPS=()
if [ -n "${ALLOWED_HOSTS:-}" ]; then
    IFS=',' read -ra HOSTS <<< "$ALLOWED_HOSTS"
    for raw_host in "${HOSTS[@]}"; do
        host=$(printf '%s' "$raw_host" | sed 's|.*://||; s|[:/].*||' | tr -d '[:space:]')
        [ -z "$host" ] && continue
        while read -r ip _; do
            [ -n "$ip" ] && ALLOWED_IPS+=("$ip")
        done < <(getent ahostsv4 "$host" 2>/dev/null | sort -u || true)
    done
fi

iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT
iptables -A OUTPUT -d 192.168.0.0/16 -j ACCEPT
iptables -A OUTPUT -d 127.0.0.11 -j ACCEPT
for ip in "${ALLOWED_IPS[@]}"; do
    iptables -A OUTPUT -d "$ip" -j ACCEPT
done
iptables -A OUTPUT -j REJECT --reject-with icmp-net-unreachable

if command -v ip6tables >/dev/null 2>&1; then
    ip6tables -F OUTPUT 2>/dev/null || true
    ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
    ip6tables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
    ip6tables -A OUTPUT -j REJECT 2>/dev/null || true
fi

echo "cybergym-network: restricted runtime egress enabled (${#ALLOWED_IPS[@]} explicit IPs)"
exec "$@"
