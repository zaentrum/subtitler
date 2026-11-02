-- The backlog of `subtitler ocr` (deploy/ocr-job.yaml): the catalog's titles with an image
-- subtitle (PGS, VobSub) in a language of the instance and no text subtitle of its kind (full,
-- SDH, forced), newest first, as titles.json. Read-only; run it in a read-only transaction:
--
--   psql -X -At -v ON_ERROR_STOP=1 -v langs="'en','de','zh'" -f ocr-titles.sql katalog > titles.json
--
-- It reads the catalog's rows as `subtitler ocr` reads package.json: the kind from the forced flag
-- and the label (the rows carry SDH in their label alone), Chinese of the script its label names
-- (either, where it names none), commentary and written Cantonese left out. A title that has a copy
-- attached already (origin `recognized`) of that language and kind is left out, so a run after a
-- run takes the rest; a subtitle a model made (translated, transcribed) covers no image subtitle.
-- Its last word is the run's: it reads package.json and plans each title again.
SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;
WITH s AS (
  SELECT a.item_id, a.path, a.format, a.origin,
         CASE lower(a.lang) WHEN 'eng' THEN 'en' WHEN 'ger' THEN 'de' WHEN 'deu' THEN 'de' WHEN 'chi' THEN 'zh'
              WHEN 'zho' THEN 'zh' WHEN 'cmn' THEN 'zh' WHEN 'fre' THEN 'fr' WHEN 'fra' THEN 'fr' WHEN 'spa' THEN 'es'
              WHEN 'ita' THEN 'it' WHEN 'dut' THEN 'nl' WHEN 'nld' THEN 'nl' WHEN 'por' THEN 'pt' WHEN 'swe' THEN 'sv'
              WHEN 'dan' THEN 'da' WHEN 'nor' THEN 'no' WHEN 'nob' THEN 'no' WHEN 'fin' THEN 'fi' WHEN 'pol' THEN 'pl'
              WHEN 'rus' THEN 'ru' WHEN 'jpn' THEN 'ja' WHEN 'kor' THEN 'ko' ELSE lower(a.lang) END AS lang,
         CASE WHEN a.isforced OR coalesce(a.label, '') ~* '\mforced\M|foreign (language )?parts' THEN 'forced'
              WHEN a.issdh OR coalesce(a.label, '') ~* '\msdh\M|hearing|\mcc\M|\[cc\]' THEN 'sdh' ELSE 'full' END AS kind,
         CASE WHEN coalesce(a.label, '') ~* 'simplified|简体|簡體|\mchs\M|zh.?hans|zh.?cn' THEN 'Hans'
              WHEN coalesce(a.label, '') ~* 'traditional|繁體|繁体|\mcht\M|zh.?hant|zh.?tw|zh.?hk|hong ?kong|taiwan'
              THEN 'Hant' END AS script,
         coalesce(a.label, '') ~* 'cantonese|粵語|粤语|\myue\M' AS cantonese,
         coalesce(a.label, '') ~* 'comment|kommentar' AS commentary,
         a.format IN ('pgs', 'vobsub') AS bitmap
  FROM com_nalet_katalog_subtitleassets a),
b AS (SELECT * FROM s WHERE bitmap AND origin IS NULL AND lang IN (:langs) AND NOT commentary AND NOT cantonese),
todo AS (
  SELECT DISTINCT b.item_id, regexp_replace(b.path, '/subs/[^/]+$', '') AS package
  FROM b
  WHERE NOT EXISTS (SELECT 1 FROM s t WHERE t.item_id = b.item_id AND NOT t.bitmap AND t.lang = b.lang
                    AND t.kind = b.kind AND NOT t.commentary AND (t.origin IS NULL OR t.origin = 'recognized')
                    AND (b.lang <> 'zh' OR b.script IS NULL OR t.script IS NULL OR t.script = b.script)))
SELECT coalesce(json_agg(json_build_object('id', t.item_id, 'name', t.item_id, 'package', t.package)
                         ORDER BY i.createdat DESC NULLS LAST, t.item_id), '[]')
FROM todo t JOIN com_nalet_katalog_items i ON i.id = t.item_id;
