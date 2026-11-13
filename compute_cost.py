#!/usr/bin/env python3
"""Cost per 1,000 requests from measured throughput (results/latency_t1.json) and the Claude
Code CLI's own list-price estimate (results/claude_calls_t1.json). Every rate below is an
ASSUMPTION, not a quoted market price.

  python compute_cost.py            writes results/cost_t1.json and prints a Markdown table
"""
import json

ENERGY_EUR_PER_KWH = 0.30          # assumption
RENTAL_EUR_PER_GPU_HOUR = [0.25, 0.50]   # illustrative assumptions
USD_TO_EUR = 1.0                   # assumption: parity, used only for the break-even volume
CLAUDE_LABEL = ("the Claude Code CLI's list-price estimate with 25 messages per call and prompt "
                "caching; not a measured API bill, and not a charge on a subscription")


def main():
    lat = json.load(open("results/latency_t1.json"))
    calls = json.load(open("results/claude_calls_t1.json"))["calls"]
    mps = lat["throughput"]["16"]["messages_per_second"]
    watts = lat["power_during_batch16"]["mean_watts"]
    gpu_hours_per_1k = 1000 / mps / 3600

    rates = [{"name": f"energy only: measured {watts:.0f} W at EUR {ENERGY_EUR_PER_KWH}/kWh",
              "eur_per_gpu_hour": watts / 1000 * ENERGY_EUR_PER_KWH, "kind": "assumption"}]
    rates += [{"name": f"illustrative rental at EUR {r:.2f}/GPU-hour", "eur_per_gpu_hour": r,
               "kind": "assumption"} for r in RENTAL_EUR_PER_GPU_HOUR]

    n_msgs = 25 * len({c["batch"] for c in calls})
    claude_total_usd = sum(c["cli_list_price_estimate_usd"] or 0 for c in calls)
    claude_usd_per_1k = claude_total_usd / n_msgs * 1000
    claude_eur_per_request = claude_usd_per_1k / 1000 * USD_TO_EUR

    for r in rates:
        r["eur_per_1000_requests"] = gpu_hours_per_1k * r["eur_per_gpu_hour"]
        # A dedicated GPU costs its hourly rate whatever the traffic; the small model is cheaper
        # only when the Claude cost of the hour's requests exceeds that rate.
        r["break_even_requests_per_hour"] = r["eur_per_gpu_hour"] / claude_eur_per_request
        r["max_requests_per_hour_at_batch16"] = mps * 3600

    out = {
        "small_model": {"throughput_batch16_messages_per_second": mps,
                        "gpu_hours_per_1000_requests": gpu_hours_per_1k,
                        "measured_mean_power_watts_batch16": watts,
                        "gpu": lat["gpu"], "shared_gpu": lat["shared_gpu"],
                        "rates": rates},
        "claude": {"usd_per_1000_requests": claude_usd_per_1k, "messages": n_msgs,
                   "total_usd_for_messages": claude_total_usd, "label": CLAUDE_LABEL},
        "assumptions": {"energy_eur_per_kwh": ENERGY_EUR_PER_KWH,
                        "rental_eur_per_gpu_hour": RENTAL_EUR_PER_GPU_HOUR,
                        "usd_to_eur": USD_TO_EUR,
                        "usd_to_eur_note": "parity is a simplification used only to put both "
                                           "costs in one currency for the break-even volume; "
                                           "scale the break-even by your own exchange rate",
                        "note": "None of the hourly rates is a quoted market price. Energy-only "
                                "excludes hardware, hosting and staff. The measured power is whole-GPU "
                                "draw on a shared card."},
        "break_even_logic": ("A dedicated GPU costs the same whether it handles 10 or 10,000 requests "
                             "an hour, so the small model is cheaper only above the break-even "
                             "volume = hourly GPU rate / Claude cost per request."),
    }
    with open("results/cost_t1.json", "w") as f:
        json.dump(out, f, indent=1)

    print("| System | Assumed rate | Cost per 1,000 requests | Break-even volume |")
    print("|---|---|---|---|")
    for r in rates:
        print(f"| Qwen3-4B + LoRA, batch 16 ({mps} msg/s) | {r['name']} (EUR {r['eur_per_gpu_hour']:.3f}/h) | "
              f"EUR {r['eur_per_1000_requests']:.4f} | {r['break_even_requests_per_hour']:,.0f} requests/hour |")
    print(f"| Claude via Claude Code CLI | {CLAUDE_LABEL} | USD {claude_usd_per_1k:.2f} | n/a |")
    print(f"\nThe small model's maximum measured throughput at batch 16 is {mps * 3600:,.0f} requests/hour.")


if __name__ == "__main__":
    main()
