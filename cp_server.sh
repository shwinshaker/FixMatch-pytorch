

paths=(
        'cifar10@4000_mean_teacher_reg=sparse_sparsity=0.1'
        'cifar10@4000_mean_teacher_reg=sparse_sparsity=0.3'
        'cifar10@4000_mean_teacher_reg=sparse_sparsity=0.5'
)

dir_src='results_fifth'
dir_dest='results'

for path in "${paths[@]}"; do
    echo
    echo '==>' $path
    cp -rs "$(readlink -f $dir_src)/$path" "$dir_dest"
done


