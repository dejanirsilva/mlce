function pullback_manual(x1, x2)
    v1, v2 = x1, x2
    v3 = v1 + v2
    v4 = v1^2
    v5 = v3 * v4
    function back(v̅5)
        v̅3 = v̅5 * v4
        v̅4 = v̅5 * v3
        v̅1 = v̅3 * 1 + v̅4 * (2v1)
        v̅2 = v̅3 * 1
        return (v̅1, v̅2)
    end
    return v5, back
end
