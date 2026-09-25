#ifndef MINI_JSON_H
#define MINI_JSON_H
// 极简 JSON 解析/序列化, 只覆盖本项目的 info.json 需求(对象/数组/字符串/数字/布尔/null)。
// 之所以不引入第三方库: 本仓库离线构建, 且 info.json 的 schema 完全由本项目控制。

#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdint>

namespace minijson {

class Value {
public:
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type type = NUL;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> members;

    bool is_null() const { return type == NUL; }
    bool is_num()  const { return type == NUM; }
    bool is_str()  const { return type == STR; }
    bool is_arr()  const { return type == ARR; }
    bool is_obj()  const { return type == OBJ; }

    const Value* find(const std::string& key) const {
        if (type != OBJ) return nullptr;
        for (const auto& kv : members)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    // 缺字段时返回 NUL 值, 让调用方用默认值兜底
    const Value& get(const std::string& key) const {
        static const Value nul;
        const Value* v = find(key);
        return v ? *v : nul;
    }
    double as_num() const {
        if (type != NUM) throw std::runtime_error("JSON 类型错误: 期望数字");
        return number;
    }
    int as_int() const { return static_cast<int>(as_num()); }
    const std::string& as_str() const {
        if (type != STR) throw std::runtime_error("JSON 类型错误: 期望字符串");
        return str;
    }
};

// ---------------- 解析 ----------------
class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value parse() {
        Value v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) fail("JSON 末尾有多余字符");
        return v;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON 解析错误(偏移 " + std::to_string(pos_) + "): " + msg);
    }
    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }
    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) fail("意外的输入结束");
        return s_[pos_];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("期望字符 '") + c + "'");
        ++pos_;
    }

    Value parse_value() {
        char c = peek();
        switch (c) {
        case '{': return parse_object();
        case '[': return parse_array();
        case '"': { Value v; v.type = Value::STR; v.str = parse_string(); return v; }
        case 't': case 'f': {
            Value v; v.type = Value::BOOL;
            if (s_.compare(pos_, 4, "true") == 0) { v.boolean = true; pos_ += 4; }
            else if (s_.compare(pos_, 5, "false") == 0) { v.boolean = false; pos_ += 5; }
            else fail("非法字面量");
            return v;
        }
        case 'n':
            if (s_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Value(); }
            fail("非法字面量");
        default:
            return parse_number();
        }
    }

    Value parse_object() {
        expect('{');
        Value v; v.type = Value::OBJ;
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            std::string key = parse_string();
            expect(':');
            v.members.emplace_back(std::move(key), parse_value());
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; break; }
            fail("对象中期望 ',' 或 '}'");
        }
        return v;
    }

    Value parse_array() {
        expect('[');
        Value v; v.type = Value::ARR;
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; break; }
            fail("数组中期望 ',' 或 ']'");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("字符串未闭合");
            char c = s_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= s_.size()) fail("转义序列未闭合");
                char e = s_[pos_++];
                switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("\\u 转义不完整");
                    unsigned cp = std::stoul(s_.substr(pos_, 4), nullptr, 16);
                    pos_ += 4;
                    // 代理对
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= s_.size() &&
                        s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                        unsigned lo = std::stoul(s_.substr(pos_ + 2, 4), nullptr, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            pos_ += 6;
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("非法转义字符");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    Value parse_number() {
        skip_ws();
        size_t start = pos_;
        while (pos_ < s_.size() &&
               (std::isdigit((unsigned char)s_[pos_]) || s_[pos_] == '+' || s_[pos_] == '-' ||
                s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E'))
            ++pos_;
        if (start == pos_) fail("期望一个值");
        Value v; v.type = Value::NUM;
        try {
            v.number = std::stod(s_.substr(start, pos_ - start));
        } catch (...) {
            fail("非法数字");
        }
        return v;
    }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

// ---------------- 序列化 ----------------
inline std::string escape_string(const std::string& s) {
    std::ostringstream oss;
    oss << '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"':  oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                oss << buf;
            } else {
                oss << char(c);
            }
        }
    }
    oss << '"';
    return oss.str();
}

// 整数写整数形式, 浮点写足够精度的小数形式 (与 Python json.dump 观感一致)
inline std::string dump_number(double v) {
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
        return std::to_string(static_cast<long long>(v));
    }
    std::ostringstream oss;
    oss.precision(17);
    oss << v;
    return oss.str();
}

} // namespace minijson

#endif // MINI_JSON_H
