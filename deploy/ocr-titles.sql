-- The backlog of `subtitler ocr` (deploy/ocr-job.yaml): the catalog's titles with an image
-- subtitle (PGS, VobSub) in a language of the instance and no text subtitle of its kind, newest
-- first, as titles.json. Read-only, and quiet so that the output is the JSON alone:
--
--   PGOPTIONS='-c default_transaction_read_only=on' \
--     psql -X -q -At -v ON_ERROR_STOP=1 -v langs="'en','de','zh'" -f ocr-titles.sql katalog > titles.json
--
-- It reads the catalog's rows as `subtitler ocr` reads package.json:
-- - the language by its primary subtag (ger, deu and de are de; a copy attached as zh-Hans is zh),
--   Chinese of the script its tag or its label names (either, where neither does);
-- - the kind: signs where the label says so, forced by the flag or the label, SDH by the label (a
--   package's rows carry SDH in their label alone), else full; commentary and written Cantonese
--   left out;
-- - an image subtitle of the version the title plays (complete, not superseded), in its folder's
--   subs/; one entry a title: its version folder and the version's id (what `subtitler attach`
--   checks the title still plays);
-- - a text subtitle of the same language and kind covers it (a forced one signs too: both are
--   forced): the package's own, and one attached that came from the title's file or elsewhere
--   (source, provider) or that a run before made (recognized), so a run after a run takes the rest;
--   one a model made (translated, transcribed) and commentary cover none.
-- Its last word is the run's: it reads package.json and plans each title again, and the attach
-- step refuses a second copy of a language and kind.
WITH s AS (
  SELECT a.item_id, a.path, a.origin,
         CASE split_part(replace(lower(coalesce(a.lang, '')), '_', '-'), '-', 1)
              WHEN 'eng' THEN 'en' WHEN 'ger' THEN 'de' WHEN 'deu' THEN 'de' WHEN 'chi' THEN 'zh' WHEN 'zho' THEN 'zh'
              WHEN 'cmn' THEN 'zh' WHEN 'fre' THEN 'fr' WHEN 'fra' THEN 'fr' WHEN 'spa' THEN 'es' WHEN 'ita' THEN 'it'
              WHEN 'dut' THEN 'nl' WHEN 'nld' THEN 'nl' WHEN 'por' THEN 'pt' WHEN 'swe' THEN 'sv' WHEN 'dan' THEN 'da'
              WHEN 'nor' THEN 'no' WHEN 'nob' THEN 'no' WHEN 'fin' THEN 'fi' WHEN 'pol' THEN 'pl' WHEN 'rus' THEN 'ru'
              WHEN 'jpn' THEN 'ja' WHEN 'kor' THEN 'ko'
              ELSE split_part(replace(lower(coalesce(a.lang, '')), '_', '-'), '-', 1) END AS lang,
         CASE WHEN coalesce(a.label, '') ~* '\msigns?\M' THEN 'signs'
              WHEN a.isforced OR coalesce(a.label, '') ~* '\mforced\M|foreign (language )?parts' THEN 'forced'
              WHEN a.issdh OR coalesce(a.label, '') ~* '\msdh\M|hearing|\mcc\M|\[cc\]' THEN 'sdh' ELSE 'full' END AS kind,
         CASE WHEN lower(coalesce(a.lang, '')) ~ '^(zh|zho|chi|cmn)[-_](hans|cn|sg)($|[-_])' THEN 'Hans'
              WHEN lower(coalesce(a.lang, '')) ~ '^(zh|zho|chi|cmn)[-_](hant|tw|hk|mo)($|[-_])' THEN 'Hant'
              WHEN coalesce(a.label, '') ~* 'simplified|简体|簡體|\mchs\M|zh.?hans|zh.?cn' THEN 'Hans'
              WHEN coalesce(a.label, '') ~* 'traditional|繁體|繁体|\mcht\M|zh.?hant|zh.?tw|zh.?hk|hong ?kong|taiwan'
              THEN 'Hant' END AS script,
         coalesce(a.label, '') ~* 'cantonese|粵語|粤语|\myue\M' AS cantonese,
         coalesce(a.label, '') ~* 'comment|kommentar' AS commentary,
         lower(coalesce(a.format, '')) IN ('pgs', 'vobsub') AS bitmap
  FROM com_nalet_katalog_subtitleassets a),
b AS (
  SELECT s.*, v.id AS version, v.dir, v.completedat
  FROM s JOIN com_nalet_katalog_itemversions v
    ON v.id = substring(s.path from '/versions/([^/]+)/subs/[^/]+$') AND v.item_id = s.item_id
  WHERE s.bitmap AND s.origin IS NULL AND s.lang IN (:langs) AND NOT s.commentary AND NOT s.cantonese
    AND v.state = 'complete' AND v.supersededby IS NULL AND v.dir IS NOT NULL),
todo AS (
  SELECT DISTINCT ON (b.item_id) b.item_id, b.dir AS package, b.version
  FROM b
  WHERE NOT EXISTS (SELECT 1 FROM s t WHERE t.item_id = b.item_id AND NOT t.bitmap AND t.lang = b.lang
                    AND (t.kind = b.kind OR (b.kind = 'signs' AND t.kind = 'forced')) AND NOT t.commentary
                    AND (t.origin IS NULL OR t.origin IN ('source', 'provider', 'recognized'))
                    AND (b.lang <> 'zh' OR b.script IS NULL OR t.script IS NULL OR t.script = b.script))
  ORDER BY b.item_id, b.completedat DESC NULLS LAST)
SELECT coalesce(json_agg(json_build_object('id', t.item_id, 'package', t.package, 'version', t.version)
                         ORDER BY i.createdat DESC NULLS LAST, t.item_id), '[]')
FROM todo t JOIN com_nalet_katalog_items i ON i.id = t.item_id;
