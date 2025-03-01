
def seconds_to_str(seconds):
    # turn seconds to days, hours, minutes
    days = int(seconds // 86400)
    seconds = seconds % 86400
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    mins = int(seconds // 60)
    return f"{days}d{hours}h{mins}m"