 set ORIGIN=data\lkp_synthesize\origin
 set FILTER=data\\lkp_synthesize\\filter

 
if not exist "%ORIGIN%" mkdir "%ORIGIN%"
if not exist "%FILTER%" mkdir "%FILTER%"


  :: ── 2wayfile ──────────────────────────────────────────────────────────────                
@REM   python build_tangled_dataset.py --mode build   --strategy 1 0 0 0 0 0 0 --out %ORIGIN%\tangled_2wayfile.json --max-build-stmts 10000
@REM   python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_2wayfile.json --out-dir %ORIGIN%
  @REM python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_2wayfile.json --out %FILTER%\tangled_2wayfile_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts 4000
  @REM python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_2wayfile_filtered.json --out-dir %FILTER%

  :: ── 2wayfunction ──────────────────────────────────────────────────────────
@REM   python build_tangled_dataset.py --mode build   --strategy 0 1 0 0 0 0 0 --out %ORIGIN%\tangled_2wayfunction.json --max-build-stmts 10000
@REM   python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_2wayfunction.json --out-dir %ORIGIN%
  @REM python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_2wayfunction.json --out %FILTER%\tangled_2wayfunction_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts  4000
  @REM python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_2wayfunction_filtered.json --out-dir %FILTER%

  :: ── connected_file ────────────────────────────────────────────────────────
@REM   python build_tangled_dataset.py --mode build   --strategy 0 0 1 0 0 0 0 --out %ORIGIN%\tangled_connected_file.json --max-build-stmts 10000
@REM   python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_connected_file.json --out-dir %ORIGIN%
  @REM python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_connected_file.json --out %FILTER%\tangled_connected_file_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts   4000
  @REM python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_connected_file_filtered.json --out-dir %FILTER%

  :: ── connected_func ────────────────────────────────────────────────────────
@REM   python build_tangled_dataset.py --mode build   --strategy 0 0 0 1 0 0 0 --out %ORIGIN%\tangled_connected_func.json --max-build-stmts 10000
  @REM python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_connected_func.json --out-dir %ORIGIN%
  @REM python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_connected_func.json --out %FILTER%\tangled_connected_func_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts   4000
  @REM python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_connected_func_filtered.json --out-dir %FILTER%

  :: ── all_file ──────────────────────────────────────────────────────────────
  @REM python build_tangled_dataset.py --mode build   --strategy 0 0 0 0 1 0 0 --out %ORIGIN%\tangled_all_file.json --max-build-stmts 10000
  @REM python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_all_file.json --out-dir %ORIGIN%
  @REM python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_all_file.json --out %FILTER%\tangled_all_file_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts 4000
  @REM python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_all_file_filtered.json --out-dir %FILTER%

  :: ── all_func ──────────────────────────────────────────────────────────────
  @REM python build_tangled_dataset.py --mode build   --strategy 0 0 0 0 0 1 0 --out %ORIGIN%\tangled_all_func.json --max-build-stmts 10000
  @REM python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_all_func.json --out-dir %ORIGIN%
  python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_all_func.json --out %FILTER%\tangled_all_func_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts 4000
  python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_all_func_filtered.json --out-dir %FILTER%

  @REM :: ── all_direct ────────────────────────────────────────────────────────────
  @REM python build_tangled_dataset.py --mode build   --strategy 0 0 0 0 0 0 1 --out %ORIGIN%\tangled_all_direct.json --max-build-stmts 5000
  @REM python build_tangled_dataset.py --mode analyze --input %ORIGIN%\tangled_all_direct.json --out-dir %ORIGIN%
  python build_tangled_dataset.py --mode filter  --input %ORIGIN%\tangled_all_direct.json --out %FILTER%\tangled_all_direct_filtered.json --min-atoms 2 --max-atoms 10 --min-stmts 2 --max-stmts 4000
  python build_tangled_dataset.py --mode analyze --input %FILTER%\tangled_all_direct_filtered.json --out-dir %FILTER%