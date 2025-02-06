# FEniCS における計算式と得られる形状

## **1. スカラー (`a`), ベクトル (`\vec{v}`), 行列 (`\boldsymbol{M}`) の定義**
- **スカラー**: $a, b, c$  （例: `Constant(1.0)`, `FunctionSpace` 内の `Function`）
- **ベクトル**: $\vec{v}, \vec{w}$  （例: `VectorFunctionSpace` の `Function`）
- **行列**: $\boldsymbol{M}, \boldsymbol{K}$  （例: `assemble()` で生成された `PETScMatrix`）

---

## **2. 計算式と得られる形状**

| **計算式** | **数式表現** | **得られる形状** | **説明** |
|------------|------------|--------------|--------------|
| `a * b` | $a \cdot b$ | **スカラー** | スカラー同士の積 |
| `a * v` | $a \vec{v}$ | **ベクトル** | スカラーとベクトルの積 |
| `a * M` | $a \boldsymbol{M}$ | **行列** | スカラーと行列の積 |
| `v + w` | $\vec{v} + \vec{w}$ | **ベクトル** | ベクトルの加算 |
| `v - w` | $\vec{v} - \vec{w}$ | **ベクトル** | ベクトルの減算 |
| `\text{dot}(v, w)` | $\vec{v} \cdot \vec{w}$ | **スカラー** | ベクトルの内積 |
| `\text{cross}(v, w)` | $\vec{v} \times \vec{w}$ | **ベクトル** | ベクトルの外積 (3D のみ) |
| `\text{inner}(v, w)` | $\vec{v}^T \vec{w}$ | **スカラー** | ベクトルの内積（`dot()` と同じ） |
| `M * v` | $\boldsymbol{M} \vec{v}$ | **ベクトル** | 行列とベクトルの積 |
| `M * M` | $\boldsymbol{M} \boldsymbol{M}$ | **行列** | 行列同士の積 |
| `v^T * M` | $\vec{v}^T \boldsymbol{M}$ | **ベクトル** | ベクトルの転置と行列の積 |
| `v^T * w` | $\vec{v}^T \vec{w}$ | **スカラー** | ベクトルの転置とベクトルの積 |
| `\text{inv}(M)` | $\boldsymbol{M}^{-1}$ | **行列** | 行列の逆行列（`solve()` で解く） |
| `\text{tr}(M)` | $\text{tr}(\boldsymbol{M})$ | **スカラー** | 行列のトレース |
| `\text{det}(M)` | $\det(\boldsymbol{M})$ | **スカラー** | 行列の行列式 |
| `\text{norm}(v, "l2")` | $\|\vec{v}\|_2$ | **スカラー** | ベクトルの L2 ノルム |
| `\text{norm}(M, "frobenius")` | $\|\boldsymbol{M}\|_F$ | **スカラー** | 行列のフロベニウスノルム |

---

## **3. 補足**
- `\text{dot}(v, w)` と `\text{inner}(v, w)` は **スカラーを返す** 。
- `M * v` は **ベクトルを返す** 。
- `M * M` は **行列を返す** 。
- `\text{cross}(v, w)` は **3D のみで有効** 。
- `\text{inv}(M)` は **直接計算するのではなく、`solve(A, x, b)` を使用するのが推奨** 。

---
