use criterion::{criterion_group, criterion_main, Criterion};
use mpf::grid::{fit, TreeGridParams};
#[path = "../tests/test_data.rs"]
mod test_data;
use rand::thread_rng;
use test_data::setup_data_csv;

fn bench_tree_grid_fitter(c: &mut Criterion) {
    let (x, y) = setup_data_csv();

    let mut group = c.benchmark_group("TreeGridFitter");

    group.bench_function("OptimizedFitter", |b| {
        b.iter(|| {
            let mut rng = thread_rng();
            fit(x.view(), y.view(), &TreeGridParams::default(), &mut rng);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_tree_grid_fitter);
criterion_main!(benches);
