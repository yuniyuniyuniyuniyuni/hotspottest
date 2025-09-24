import os
import pandas as pd

# === 설정 ==============================================================
BASE_PATH = "2022_1/"

# 업로드된 실제 파일명과 일치해야 합니다.
FNAME = {
    "유동인구":   "서울시 상권분석서비스(길단위인구-행정동)_20221.csv",
    "상권변화":   "서울시 상권분석서비스(상권변화지표-행정동)_20221.csv",
    "상주인구":   "서울시 상권분석서비스(상주인구-행정동)_20221.csv",
    "소득소비":   "서울시 상권분석서비스(소득소비-행정동)_20221.csv",
    "아파트":     "서울시 상권분석서비스(아파트-행정동)_20221.csv",
    "직장인구":   "서울시 상권분석서비스(직장인구-행정동)_20221.csv",
    "집객시설":   "서울시 상권분석서비스(집객시설-행정동)_20221.csv",
    "점포":       "서울시_상권분석서비스(점포-행정동)_20221_외식업.csv",
    "추정매출":   "서울시_상권분석서비스(추정매출-행정동)_20221_외식업.csv",
}

# 키 컬럼
DONG_KEYS = ["기준_년분기_코드", "행정동_코드", "행정동_코드_명"]
FULL_KEYS = ["기준_년분기_코드", "행정동_코드", "서비스_업종_코드"]

# 결과 파일 경로
OUT_FULLGRID = os.path.join(BASE_PATH, "행정동X업종_통합_20221.csv")
OUT_MISSING  = os.path.join(BASE_PATH, "누락조합_리스트_20221.csv")
# ======================================================================

def read_csv(alias: str) -> pd.DataFrame:
    """별칭(alias)으로 CSV 읽기 (UTF-8)."""
    path = os.path.join(BASE_PATH, FNAME[alias])
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    return pd.read_csv(path, encoding="utf-8")

def add_prefix(df: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    """키를 제외한 컬럼에 접두어(prefix_) 부여."""
    non_keys = [c for c in df.columns if c not in keys]
    rename_map = {c: f"{prefix}_{c}" for c in non_keys}
    return df.rename(columns=rename_map)

def attach_block(base: pd.DataFrame, alias: str, prefix: str | None = None) -> pd.DataFrame:
    """행정동 단위(업종X) 블록을 LEFT JOIN으로 부착."""
    df = read_csv(alias)
    if prefix is None:
        prefix = alias
    non_keys = [c for c in df.columns if c not in DONG_KEYS]
    rename_map = {c: f"{prefix}_{c}" for c in non_keys}
    df_renamed = df.rename(columns=rename_map)
    return base.merge(df_renamed, on=DONG_KEYS, how="left")

def main():
    # ---------- 1) 행정동 목록(425개) & 업종(10개)로 '완전 그리드' 생성 ----------
    dong_df = read_csv("유동인구")[DONG_KEYS].drop_duplicates()
    dongs = dong_df[["기준_년분기_코드", "행정동_코드"]].drop_duplicates()
    dong_name_map = dong_df.drop_duplicates(subset=["행정동_코드"])[["행정동_코드", "행정동_코드_명"]]

    shop_keys  = read_csv("점포")[FULL_KEYS + ["서비스_업종_코드_명"]].drop_duplicates()
    sales_keys = read_csv("추정매출")[FULL_KEYS + ["서비스_업종_코드_명"]].drop_duplicates()

    # 업종 목록(10개) 도출
    categories = pd.concat(
        [
            shop_keys[["서비스_업종_코드", "서비스_업종_코드_명"]],
            sales_keys[["서비스_업종_코드", "서비스_업종_코드_명"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    # 카티전 곱 (행정동 × 업종)
    full_grid = dongs.merge(categories, how="cross").merge(dong_name_map, on="행정동_코드", how="left")
    full_grid = full_grid[
        ["기준_년분기_코드", "행정동_코드", "행정동_코드_명", "서비스_업종_코드", "서비스_업종_코드_명"]
    ]

    # ---------- 2) 업종 단위 데이터(점포/추정매출) LEFT JOIN ----------
    shop_df  = add_prefix(read_csv("점포").copy(),  FULL_KEYS, "점포")
    sales_df = add_prefix(read_csv("추정매출").copy(), FULL_KEYS, "매출")

    merged = full_grid.merge(shop_df,  on=FULL_KEYS, how="left")
    merged = merged.merge(sales_df, on=FULL_KEYS, how="left")

    # ---------- 3) 행정동 단위 데이터 LEFT JOIN (행 개수 고정) ----------
    for alias in ["유동인구", "상주인구", "직장인구", "소득소비", "아파트", "집객시설", "상권변화"]:
        merged = attach_block(merged, alias, prefix=alias)

    # ---------- 4) 숫자형 결측치(NaN) -> 0 ----------
    num_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    merged[num_cols] = merged[num_cols].fillna(0)

    # ---------- 5) 정렬 & 저장 ----------
    merged = merged.sort_values(FULL_KEYS).reset_index(drop=True)
    merged.to_csv(OUT_FULLGRID, index=False, encoding="utf-8-sig")

    # ---------- 6) 관측이 없던(원래 빠져 있던) 조합 리포트 ----------
    observed = pd.concat([shop_keys[FULL_KEYS], sales_keys[FULL_KEYS]], ignore_index=True).drop_duplicates()
    missing = full_grid.merge(observed, on=FULL_KEYS, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    missing.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

    # ---------- 7) 요약 출력 ----------
    print("=== 통합 완료 ===")
    print(f"그리드 기대 행수(행정동×업종): {len(full_grid):,}")
    print(f"결과 행수(그리드 고정):        {len(merged):,}")
    print(f"숫자형 컬럼 수(NaN→0):        {len(num_cols)}개")
    print(f"관측 누락 조합 수:            {len(missing):,}")
    print(f"통합 파일: {OUT_FULLGRID}")
    print(f"누락조합 파일: {OUT_MISSING}")

if __name__ == "__main__":
    # 파일 존재 여부 간단 체크
    missing_files = [name for name in FNAME.values() if not os.path.exists(os.path.join(BASE_PATH, name))]
    if missing_files:
        raise FileNotFoundError(f"다음 파일이 존재하지 않습니다:\n- " + "\n- ".join(missing_files))
    main()
