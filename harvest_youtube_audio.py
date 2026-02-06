import subprocess
import os

def download_audio(urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            url
        ]
        subprocess.run(command)

# 🔫 Gaming (no commentary)
gaming_urls = [
    
    "https://youtu.be/ZQgtG9D2JVI?si=-un6XMFJIRAOpJVh",
    "https://youtu.be/jq0Mp5ZscS4?si=d5C3aS-uyKUQGGFW",
    "https://youtu.be/KkKizl7ExIU?si=kdaatTXPFehTY5xX",
    "https://youtu.be/cUPShEGibHw?si=9dLbfx3NybySsHDt",
    "https://youtu.be/VXAtvYqL_vk?si=neLxj8EqpqJVyeT9",
    "https://youtu.be/Tnuf02YXvu0?si=hg7ibGIpS1jDdJRT",
    "https://youtu.be/LXk3702OIe8?si=DNq05J6cwoKE5-ty",
    "https://youtu.be/_R2DPAtO33c?si=wDJrX602xuObPjvB",
    "https://youtu.be/KkKizl7ExIU?si=bpHySmjVioD_dpfD",
    "https://youtu.be/Tnuf02YXvu0?si=6vbBeLzr04HLDcPX",
    "https://youtu.be/UF4Cws8Rjlw?si=5T8Ti8IP-AsUk44W",
    "https://youtu.be/_R2DPAtO33c?si=ajTadeps31DS9gMG",
    "https://youtu.be/cUPShEGibHw?si=xsuw2bfwzYj7PEJ9",
    "https://youtu.be/zxLdGJh5h7Q?si=Kamx3SSjFnRY-q7r",
    "https://youtu.be/z7wsWDYG2_c?si=yUHoMwlMAqMd5WK7",
    "https://youtu.be/PmZk6CgzZGo?si=nxLJhczunVw_wBPv",
    "https://youtu.be/VcpROakdLcw?si=YgAJuIi4nN7b-gds",
    "https://youtu.be/cZiL9uk8o20?si=SE_ap5zUU0TpYmBV",
    "https://youtu.be/PVXJzjdnPAU?si=57YSy6NxhjLFF5pF",
    "https://youtu.be/2HX9VBm3xx4?si=o1C52pjNNp7qeTOH"

]

# 🎬 Movies
movie_urls = [
    "https://youtu.be/C7OQHIpDlvA?si=4dcT0hYEduXOHUeA",
    "https://youtu.be/3iH8l6dN6Ow?si=D3TiD5-7_xdeXDzx",
    "https://youtu.be/ufj66HZq07M?si=eJsslqtzdqVI3L5W",
    "https://youtu.be/SR__amDl1c8?si=9UVbI6ZbrR4Hy3CI",
    "https://youtu.be/TlJ7R1EAwhE?si=SZ3QBjf5wvL_B6-M",
    "https://youtu.be/2uZhpRvvVM4?si=nTfWToq8RHN22cDj",
    "https://youtu.be/RySK3jpIUxI?si=LQotjbA70qJpGhh8"
    "https://youtu.be/5hPtU8Jbpg0?si=hmlfjGZlNsC0AF_5",
    "https://youtu.be/JLmOkEEC9SQ?si=A5168sAjTA6zfsPZ",
    "https://youtu.be/Un0Zo34GOow?si=_WQoUbPp0cjcjCrm",
    "https://youtu.be/JLmOkEEC9SQ?si=Q93StUnMYG4Nptwe",
    "https://youtu.be/bLVKTbxPmcg?si=xXgsSZ08WpBCRMmV",
    "https://youtu.be/3oJ2WXg0E8g?si=ZmFF61xjk2oJCwlG",
    "https://youtu.be/9Tq71PiDJDk?si=B2NQYbswWLBK61PX",
    "https://youtu.be/3iH8l6dN6Ow?si=fJUoTy16GoI7Ncib",
    "https://youtu.be/nDCt6fUE9-o?si=VnJ63qSxuS9Xn8qZ",
    "https://youtu.be/SR__amDl1c8?si=edKs5iJ06_IVkJxt",
    "https://youtu.be/Xhn6AnCOtMI?si=8RNN-CguAkmUSflN"
    "https://youtu.be/LVvNgKQy1nI?si=U4Wm--klq_pbOCyv"
]

download_audio(gaming_urls, "data/raw/gaming")
download_audio(movie_urls, "data/raw/movie")

print("✅ YouTube audio harvested")
