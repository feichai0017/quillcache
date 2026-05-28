use std::env;
use std::path::{Path, PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quill_core::database::{Database, DatabaseOptions};
use quill_jit::JitOptions;
use tpchgen_cli::{Compression, OutputFormat, Table, TpchGenerator};

const DEFAULT_SCALE_FACTOR: f64 = 1.0;

const Q1: &str = r#"
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from lineitem
where l_shipdate <= date '1998-09-02'
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus
"#;

const Q3: &str = r#"
select
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    o_orderdate,
    o_shippriority
from customer, orders, lineitem
where c_mktsegment = 'BUILDING'
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < date '1995-03-15'
  and l_shipdate > date '1995-03-15'
group by l_orderkey, o_orderdate, o_shippriority
order by revenue desc, o_orderdate
limit 10
"#;

const Q6: &str = r#"
select
    sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1994-01-01'
  and l_shipdate < date '1995-01-01'
  and l_discount between 0.05 and 0.07
  and l_quantity < 24
"#;

#[derive(Debug, Clone, Copy)]
struct TpchQuery {
    name: &'static str,
    sql: &'static str,
    tables: &'static [&'static str],
    compare_jit_modes: bool,
}

#[derive(Debug, Clone, Copy)]
struct TpchMode {
    name: &'static str,
    jit: JitOptions,
}

const TPCH_QUERIES: &[TpchQuery] = &[
    TpchQuery {
        name: "q6_scan_filter_aggregate",
        sql: Q6,
        tables: &["lineitem"],
        compare_jit_modes: true,
    },
    TpchQuery {
        name: "q1_grouped_aggregate_sort",
        sql: Q1,
        tables: &["lineitem"],
        compare_jit_modes: false,
    },
    TpchQuery {
        name: "q3_join_aggregate",
        sql: Q3,
        tables: &["customer", "orders", "lineitem"],
        compare_jit_modes: false,
    },
];

fn bench_tpch(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let root = runtime
        .block_on(load_or_generate_data())
        .expect("prepare TPC-H data");
    let mut group = c.benchmark_group("tpch");
    group.sample_size(10);

    for query in TPCH_QUERIES {
        for mode in query_modes(query) {
            let db = match runtime.block_on(prepare_database(&root, query.tables, mode.jit)) {
                Ok(db) => db,
                Err(err) => {
                    eprintln!("skipping {}/{}: {}", query.name, mode.name, err);
                    continue;
                }
            };
            warmup_and_bench(&mut group, &runtime, &db, query, mode.name);
        }
    }

    group.finish();
}

fn query_modes(query: &TpchQuery) -> Vec<TpchMode> {
    if query.compare_jit_modes {
        vec![
            TpchMode {
                name: "datafusion/native",
                jit: JitOptions::disabled(),
            },
            TpchMode {
                name: "quill/mlir-jit",
                jit: JitOptions::mlir_execution(),
            },
        ]
    } else {
        vec![TpchMode {
            name: "datafusion/native",
            jit: JitOptions::disabled(),
        }]
    }
}

fn warmup_and_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
    db: &Database,
    query: &TpchQuery,
    mode: &str,
) {
    runtime
        .block_on(db.run(query.sql))
        .unwrap_or_else(|err| panic!("sql warmup {}/{} failed: {}", mode, query.name, err));
    let prepared = runtime
        .block_on(db.prepare(query.sql))
        .unwrap_or_else(|err| panic!("prepare {}/{} failed: {}", mode, query.name, err));
    runtime
        .block_on(prepared.run())
        .unwrap_or_else(|err| panic!("prepared warmup {}/{} failed: {}", mode, query.name, err));

    group.bench_function(BenchmarkId::new(format!("sql/{mode}"), query.name), |b| {
        b.iter(|| {
            black_box(
                runtime
                    .block_on(db.run(black_box(query.sql)))
                    .unwrap_or_else(|err| panic!("sql {}/{} failed: {}", mode, query.name, err)),
            )
        });
    });
    group.bench_function(
        BenchmarkId::new(format!("prepared/{mode}"), query.name),
        |b| {
            b.iter(|| {
                black_box(runtime.block_on(prepared.run()).unwrap_or_else(|err| {
                    panic!("prepared {}/{} failed: {}", mode, query.name, err)
                }))
            });
        },
    );
}

async fn load_or_generate_data() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("QUILL_TPCH_DIR").map(PathBuf::from) {
        eprintln!("using existing TPC-H parquet data at {}", path.display());
        return Ok(path);
    }

    let scale_factor = env_f64("QUILL_TPCH_SF", DEFAULT_SCALE_FACTOR)?;
    let threads = env_usize("QUILL_TPCH_GEN_THREADS")
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let root = generated_data_dir(scale_factor);
    if env_flag("QUILL_TPCH_REGENERATE") && root.exists() {
        std::fs::remove_dir_all(&root).map_err(|err| err.to_string())?;
    }
    if has_required_tables(&root) {
        eprintln!("using generated TPC-H parquet data at {}", root.display());
        return Ok(root);
    }

    std::fs::create_dir_all(&root).map_err(|err| err.to_string())?;
    eprintln!(
        "generating TPC-H parquet data with tpchgen-cli: sf={}, threads={}, dir={}",
        scale_factor,
        threads,
        root.display()
    );

    TpchGenerator::builder()
        .with_scale_factor(scale_factor)
        .with_output_dir(&root)
        .with_tables(vec![Table::Lineitem, Table::Orders, Table::Customer])
        .with_format(OutputFormat::Parquet)
        .with_parquet_compression(Compression::UNCOMPRESSED)
        .with_num_threads(threads)
        .build()
        .generate()
        .await
        .map_err(|err| err.to_string())?;

    Ok(root)
}

async fn prepare_database(
    root: &Path,
    tables: &[&str],
    jit: JitOptions,
) -> Result<Database, String> {
    let db = Database::new(DatabaseOptions {
        debug_trace: false,
        jit,
        ..Default::default()
    })
    .map_err(|err| err.to_string())?;
    for table in tables {
        let path = table_path(root, table)
            .ok_or_else(|| format!("missing parquet data for table {table} under {:?}", root))?;
        db.register_parquet(table, path.to_str().expect("utf-8 path"))
            .await
            .map_err(|err| err.to_string())?;
    }
    Ok(db)
}

fn env_f64(name: &str, default: f64) -> Result<f64, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<f64>()
            .map_err(|err| format!("invalid {name}={value:?}: {err}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(format!("invalid {name}: {err}")),
    }
}

fn env_usize(name: &str) -> Option<usize> {
    match env::var(name) {
        Ok(value) => Some(
            value
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("invalid {name}={value:?}: {err}")),
        ),
        Err(env::VarError::NotPresent) => None,
        Err(err) => panic!("invalid {name}: {err}"),
    }
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn generated_data_dir(scale_factor: f64) -> PathBuf {
    let scale = format_scale_factor(scale_factor);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchdata")
        .join(format!("tpch-sf{scale}"))
}

fn format_scale_factor(scale_factor: f64) -> String {
    let text = format!("{scale_factor:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn has_required_tables(root: &Path) -> bool {
    ["lineitem", "orders", "customer"]
        .into_iter()
        .all(|table| table_path(root, table).is_some())
}

fn table_path(root: &Path, table: &str) -> Option<PathBuf> {
    let parquet_file = root.join(format!("{table}.parquet"));
    if parquet_file.is_file() {
        return Some(parquet_file);
    }

    let table_dir = root.join(table);
    if table_dir.is_dir() {
        return Some(table_dir);
    }

    None
}

criterion_group!(benches, bench_tpch);
criterion_main!(benches);
