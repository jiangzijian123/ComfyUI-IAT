while read -r requirement; do
    pip install "$requirement" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple || echo "Failed to install $requirement, skipping..."
done < requirements.txt